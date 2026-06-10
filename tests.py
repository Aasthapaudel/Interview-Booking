import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_txt_fixed():
    mock_qdrant = MagicMock()
    mock_qdrant.get_collections.return_value.collections = []

    with patch("app.api.ingestion.qdrant", mock_qdrant):
        content = (
            b"Python is a programming language. FastAPI is a web framework. "
            b"RAG stands for Retrieval Augmented Generation. "
            b"It retrieves relevant documents before generating answers. "
            b"Redis is used for caching chat history in memory."
        )
        response = client.post(
            "/ingest",
            files={"file": ("sample.txt", io.BytesIO(content), "text/plain")},
            data={"chunking_strategy": "fixed"},
        )
    assert response.status_code == 200
    data = response.json()
    assert "doc_id" in data
    assert data["chunks_stored"] > 0
    assert data["strategy"] == "fixed"
    assert data["filename"] == "sample.txt"


def test_ingest_txt_sentence_window():
    mock_qdrant = MagicMock()
    mock_qdrant.get_collections.return_value.collections = []

    with patch("app.api.ingestion.qdrant", mock_qdrant):
        content = (
            b"Aastha is an ML engineer. She has five years of experience. "
            b"Her skills include Python and FastAPI. She has built RAG systems before."
        )
        response = client.post(
            "/ingest",
            files={"file": ("bio.txt", io.BytesIO(content), "text/plain")},
            data={"chunking_strategy": "sentence_window"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"] == "sentence_window"
    assert data["chunks_stored"] > 0


def test_ingest_unsupported_file_type():
    mock_qdrant = MagicMock()
    mock_qdrant.get_collections.return_value.collections = []

    with patch("app.api.ingestion.qdrant", mock_qdrant):
        response = client.post(
            "/ingest",
            files={"file": ("resume.docx", io.BytesIO(b"content"), "application/octet-stream")},
            data={"chunking_strategy": "fixed"},
        )
    assert response.status_code == 400


def test_chat_basic_reply():
    mock_qdrant = MagicMock()
    mock_qdrant.query_points.return_value.points = [
        MagicMock(payload={"text": "Python is widely used in machine learning."})
    ]
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Python is used for ML and data science."))
    ]
    mock_redis = MagicMock()
    mock_redis.lrange.return_value = []

    with patch("app.api.chat.qdrant", mock_qdrant), \
         patch("app.api.chat.groq_client", mock_groq), \
         patch("app.core.memory._r", mock_redis):
        response = client.post("/chat", json={
            "query": "What is Python used for?",
            "session_id": "test_session_001"
        })
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert data["session_id"] == "test_session_001"
    assert data["booking_saved"] is False


def test_chat_booking_intent():
    mock_qdrant = MagicMock()
    mock_qdrant.query_points.return_value.points = []

    booking_json = (
        '{"intent": "booking", "name": "Aastha Singh", '
        '"email": "aastha@example.com", "date": "2024-12-20", "time": "10:00 AM"}'
    )
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=booking_json))
    ]
    mock_redis = MagicMock()
    mock_redis.lrange.return_value = []

    with patch("app.api.chat.qdrant", mock_qdrant), \
         patch("app.api.chat.groq_client", mock_groq), \
         patch("app.core.memory._r", mock_redis):
        response = client.post("/chat", json={
            "query": (
                "I want to book an interview. My name is Aastha Singh, "
                "email aastha@example.com, date 2024-12-20, time 10:00 AM."
            ),
            "session_id": "test_session_booking"
        })
    assert response.status_code == 200
    data = response.json()
    assert data["booking_saved"] is True
    assert "Aastha Singh" in data["reply"]


def test_chat_multiturn_history():
    mock_qdrant = MagicMock()
    mock_qdrant.query_points.return_value.points = [
        MagicMock(payload={"text": "FastAPI is a modern Python web framework."})
    ]
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="FastAPI supports async and is very fast."))
    ]
    import json
    mock_redis = MagicMock()
    mock_redis.lrange.return_value = [
        json.dumps({"role": "user", "content": "Tell me about Python."}),
        json.dumps({"role": "assistant", "content": "Python is a versatile language."}),
    ]

    with patch("app.api.chat.qdrant", mock_qdrant), \
         patch("app.api.chat.groq_client", mock_groq), \
         patch("app.core.memory._r", mock_redis):
        response = client.post("/chat", json={
            "query": "What about FastAPI?",
            "session_id": "test_session_multiturn"
        })
    assert response.status_code == 200
    assert "reply" in response.json()
