from typing import Literal
from uuid import uuid4

import fitz  # PyMuPDF
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sqlalchemy.orm import Session

from app.config import settings
from app.core.chunker import chunk
from app.core.embedder import embed
from app.db.crud import save_chunks, save_document
from app.db.database import get_db

router = APIRouter(prefix="/ingest", tags=["Document Ingestion"])
qdrant = QdrantClient(url=settings.qdrant_url)


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_stored: int
    strategy: str


def ensure_collection() -> None:
    """Create the Qdrant collection if it does not already exist."""
    existing = [c.name for c in qdrant.get_collections().collections]
    if settings.collection_name not in existing:
        qdrant.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection: '{settings.collection_name}'")


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from a PDF or TXT file.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename used to detect file type.

    Returns:
        Extracted text as a single string.

    Raises:
        HTTPException: If the file type is not supported.
    """
    if filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    elif filename.lower().endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{filename}'. Only .pdf and .txt are accepted."
        )


@router.post("", response_model=IngestResponse)
def ingest(
    file: UploadFile = File(...),
    chunking_strategy: Literal["fixed", "sentence_window"] = Form("fixed"),
    db: Session = Depends(get_db),
) -> IngestResponse:
    """Ingest a document: extract text, chunk, embed, and store vectors + metadata.

    Args:
        file: Uploaded .pdf or .txt file.
        chunking_strategy: 'fixed' for word-window chunks,
                           'sentence_window' for sentence-group chunks.
        db: SQLAlchemy database session.

    Returns:
        IngestResponse with doc_id, filename, chunk count, and strategy used.
    """
    ensure_collection()

    file_bytes = file.file.read()
    text = extract_text(file_bytes, file.filename)
    logger.info(f"Extracted {len(text)} characters from '{file.filename}'")

    if not text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the file.")

    chunks = chunk(text, strategy=chunking_strategy)
    logger.info(f"Created {len(chunks)} chunks using strategy='{chunking_strategy}'")

    vectors = embed(chunks)

    doc_id = save_document(db, filename=file.filename, strategy=chunking_strategy)
    save_chunks(db, doc_id=doc_id, chunks=chunks)

    points = [
        PointStruct(
            id=str(uuid4()),
            vector=vectors[i],
            payload={"doc_id": doc_id, "chunk_index": i, "text": chunks[i]},
        )
        for i in range(len(chunks))
    ]
    qdrant.upsert(collection_name=settings.collection_name, points=points)
    logger.info(f"Stored {len(chunks)} vectors in Qdrant for doc_id='{doc_id}'")

    return IngestResponse(
        doc_id=doc_id,
        filename=file.filename,
        chunks_stored=len(chunks),
        strategy=chunking_strategy,
    )
