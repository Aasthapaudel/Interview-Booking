# RAG Backend

A production-quality RAG (Retrieval-Augmented Generation) backend built with FastAPI.
**Custom pipeline — no LangChain, no FAISS/Chroma, no RetrievalQAChain.**

## Features

- **Document Ingestion** — Upload `.pdf` or `.txt` files, extract text, apply two chunking strategies, generate embeddings, and store vectors in Qdrant with metadata in SQLite.
- **Conversational RAG** — Multi-turn chat backed by Redis memory, semantic retrieval from Qdrant, Groq LLM responses, and automatic interview booking detection.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `POST /ingest` | multipart/form | Upload PDF or TXT, chunk, embed, store in Qdrant + SQLite |
| `POST /chat` | JSON | Multi-turn conversational RAG with booking detection |

Interactive docs → **http://localhost:8000/docs**

## Tech Stack

| Component | Choice |
|---|---|
| Framework | FastAPI |
| LLM | Groq (`llama-3.1-8b-instant`) |
| Vector DB | Qdrant |
| Metadata DB | SQLite + SQLAlchemy |
| Chat Memory | Redis |
| Embeddings | `all-MiniLM-L6-v2` (local, 384-dim) |
| PDF Parsing | PyMuPDF |

## Chunking Strategies

| Strategy | Description |
|---|---|
| `fixed` | Overlapping 512-word windows with 64-word overlap. Best for dense docs. |
| `sentence_window` | Sliding groups of 3 sentences. Best for structured or conversational text. |

## Project Structure

```
rag_backend/
├── main.py                    # FastAPI app entry point + DB init
├── requirements.txt
├── .env.example               # copy to .env and fill in your keys
├── tests.py                   # full pytest test suite
└── app/
    ├── config.py              # settings loaded from .env via pydantic-settings
    ├── api/
    │   ├── ingestion.py       # POST /ingest — document pipeline
    │   └── chat.py            # POST /chat  — RAG + booking logic
    ├── core/
    │   ├── chunker.py         # fixed + sentence_window chunking
    │   ├── embedder.py        # sentence-transformers wrapper
    │   ├── memory.py          # Redis multi-turn session history
    │   └── prompt_builder.py  # manual prompt assembly — no chains
    └── db/
        ├── database.py        # SQLAlchemy engine + session factory
        ├── models.py          # Document, Chunk, Booking ORM models
        └── crud.py            # DB insert operations
```

## Setup

### 1. Prerequisites

- Python 3.10+
- Redis running on `localhost:6379`
- Qdrant running on `localhost:6333`

**Linux (Redis):**
```bash
sudo apt install redis-server
sudo systemctl start redis
```

**Qdrant (Linux binary):**
```bash
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-musl.tar.gz -o qdrant.tar.gz
tar -xzf qdrant.tar.gz
./qdrant
```

> For Windows / Mac instructions, see `HOW_TO_RUN.txt`.

### 2. Environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at https://console.groq.com)
```

### 3. Install & Run

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## Usage Examples

### Ingest a document

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@document.pdf" \
  -F "chunking_strategy=fixed"
```

### Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?", "session_id": "session1"}'
```

### Book an interview

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Book an interview. Name: Aastha Singh, email: a@b.com, date: 2024-12-20, time: 10:00 AM.", "session_id": "session1"}'
```

## Run Tests

```bash
pytest tests.py -v
```

## Constraints Met

- No FAISS / Chroma / LangChain / RetrievalQAChain
- No UI — pure REST API
- Modular, typed code following industry standards
- Full docstrings and type annotations on every function
