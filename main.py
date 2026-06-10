from fastapi import FastAPI
from loguru import logger

from app.api import chat, ingestion
from app.db.database import Base, engine

# create all SQLite tables on startup
Base.metadata.create_all(bind=engine)
logger.info("Database tables created / verified")

app = FastAPI(
    title="RAG Backend",
    description=(
        "A custom Retrieval-Augmented Generation (RAG) backend.\n\n"
        "**Document Ingestion API** — Upload `.pdf` or `.txt` files, extract text, "
        "apply two chunking strategies, embed and store vectors in Qdrant, "
        "and save metadata to SQLite.\n\n"
        "**Conversational RAG API** — Multi-turn chat backed by Redis, "
        "with Qdrant semantic retrieval, Groq LLM responses, and interview booking support."
    ),
    version="1.0.0",
    contact={"name": "Aastha Singh"},
    license_info={"name": "Private"},
)

app.include_router(ingestion.router)
app.include_router(chat.router)


@app.get("/health", include_in_schema=False)
def health() -> dict:
    """Internal health check — not shown in API docs."""
    return {"status": "ok"}
