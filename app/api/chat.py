import json as json_lib

from fastapi import APIRouter, Depends
from groq import Groq
from loguru import logger
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from app.config import settings
from app.core.embedder import embed
from app.core.memory import get_history, save_turn
from app.core.prompt_builder import build_prompt
from app.db.crud import save_booking
from app.db.database import get_db

router = APIRouter(prefix="/chat", tags=["Conversational RAG"])
qdrant = QdrantClient(url=settings.qdrant_url)
groq_client = Groq(api_key=settings.groq_api_key)


class ChatRequest(BaseModel):
    query: str
    session_id: str


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    booking_saved: bool = False


def call_llm(prompt: str) -> str:
    """Send the assembled prompt to Groq and return the response text.

    Args:
        prompt: Fully built prompt string.

    Returns:
        LLM response as a plain stripped string.
    """
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def detect_and_save_booking(
    answer: str,
    session_id: str,
    db: Session,
) -> tuple[str, bool]:
    """Check if the LLM response contains a booking intent JSON and save it.

    Args:
        answer: Raw LLM output string.
        session_id: Current chat session identifier.
        db: SQLAlchemy session for persisting the booking.

    Returns:
        Tuple of (final reply text, booking_saved flag).
    """
    try:
        parsed = json_lib.loads(answer)
        if parsed.get("intent") == "booking":
            required_fields = {"name", "email", "date", "time"}
            if required_fields.issubset(parsed.keys()):
                save_booking(db, session_id=session_id, data=parsed)
                confirmation = (
                    f"Booking confirmed for {parsed['name']} "
                    f"on {parsed['date']} at {parsed['time']}. "
                    f"A confirmation will be sent to {parsed['email']}."
                )
                logger.info(f"Booking saved for session='{session_id}'")
                return confirmation, True
    except (json_lib.JSONDecodeError, KeyError):
        pass
    return answer, False


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """Handle a conversational RAG query with Redis-backed multi-turn memory.

    Flow:
        1. Fetch session history from Redis.
        2. Embed the query and retrieve top-5 relevant chunks from Qdrant.
        3. Build a prompt combining context, history, and query.
        4. Call Groq LLM and get a response.
        5. Detect booking intent and persist if found.
        6. Save both turns to Redis and return the reply.

    Args:
        req: ChatRequest with query string and session_id.
        db: SQLAlchemy database session.

    Returns:
        ChatResponse with reply text, session_id, and booking_saved flag.
    """
    # step 1 — fetch multi-turn history from Redis
    history = get_history(req.session_id)
    logger.info(f"Fetched {len(history)} history turns for session='{req.session_id}'")

    # step 2 — embed query and search Qdrant
    query_vector = embed([req.query])[0]
    results = qdrant.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=5,
    ).points
    context_chunks = [r.payload["text"] for r in results]
    logger.info(f"Retrieved {len(context_chunks)} chunks for query='{req.query[:60]}'")

    # step 3 — build prompt manually
    _booking_keywords = {"book", "interview", "appointment", "schedule", "reserve"}
    query_words = set(req.query.lower().split())
    is_booking_query = bool(_booking_keywords & query_words)
    prompt = build_prompt(req.query, context_chunks, history, include_booking_instructions=is_booking_query)

    # step 4 — call LLM
    raw_answer = call_llm(prompt)
    logger.info(f"LLM reply (preview): {raw_answer[:80]}")

    # step 5 — detect booking intent ONLY if user's query mentions booking
    if is_booking_query:
        reply, booking_saved = detect_and_save_booking(raw_answer, req.session_id, db)
    else:
        reply, booking_saved = raw_answer, False

    # step 6 — persist both turns to Redis
    save_turn(req.session_id, "user", req.query)
    save_turn(req.session_id, "assistant", reply)

    return ChatResponse(reply=reply, session_id=req.session_id, booking_saved=booking_saved)
