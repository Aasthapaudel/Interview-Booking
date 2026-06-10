from uuid import uuid4

from sqlalchemy.orm import Session

from app.db.models import Booking, Chunk, Document


def save_document(db: Session, filename: str, strategy: str) -> str:
    """Insert a new document record and return its generated ID.

    Args:
        db: SQLAlchemy session.
        filename: Original uploaded filename.
        strategy: Chunking strategy used.

    Returns:
        Generated document UUID string.
    """
    doc_id = str(uuid4())
    db.add(Document(id=doc_id, filename=filename, strategy=strategy))
    db.commit()
    return doc_id


def save_chunks(db: Session, doc_id: str, chunks: list[str]) -> None:
    """Bulk insert all chunks for a document.

    Args:
        db: SQLAlchemy session.
        doc_id: Parent document ID.
        chunks: List of text chunk strings.
    """
    for i, text in enumerate(chunks):
        db.add(Chunk(id=str(uuid4()), doc_id=doc_id, chunk_index=i, text=text))
    db.commit()


def save_booking(db: Session, session_id: str, data: dict) -> None:
    """Persist an interview booking extracted from the LLM response.

    Args:
        db: SQLAlchemy session.
        session_id: Chat session that triggered the booking.
        data: Dict with keys name, email, date, time.
    """
    db.add(Booking(
        id=str(uuid4()),
        session_id=session_id,
        name=data.get("name"),
        email=data.get("email"),
        date=data.get("date"),
        time=data.get("time"),
    ))
    db.commit()
