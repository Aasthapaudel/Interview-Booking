import re
from typing import Literal


def chunk(
    text: str,
    strategy: Literal["fixed", "sentence_window"],
    chunk_size: int = 512,
    overlap: int = 64,
    window: int = 3,
) -> list[str]:
    """Split text into chunks using the selected strategy.

    Args:
        text: Raw extracted text from the document.
        strategy: 'fixed' for fixed-size word windows,
                  'sentence_window' for sliding sentence groups.
        chunk_size: Max words per chunk (fixed strategy only).
        overlap: Overlapping words between chunks (fixed strategy only).
        window: Number of sentences per chunk (sentence_window only).

    Returns:
        List of non-empty text chunks.
    """
    if strategy == "fixed":
        return _fixed_size(text, chunk_size, overlap)
    return _sentence_window(text, window)


def _fixed_size(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping fixed-size word windows.

    Args:
        text: Input text.
        size: Number of words per chunk.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        List of word-window chunks.
    """
    words = text.split()
    step = max(1, size - overlap)
    chunks: list[str] = []
    for i in range(0, len(words), step):
        piece = " ".join(words[i : i + size])
        if piece.strip():
            chunks.append(piece)
    return chunks


def _sentence_window(text: str, window: int) -> list[str]:
    """Split text into sliding windows of consecutive sentences.

    Args:
        text: Input text.
        window: Number of sentences per chunk.

    Returns:
        List of sentence-group chunks.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) <= window:
        return [" ".join(sentences)]
    return [
        " ".join(sentences[i : i + window])
        for i in range(len(sentences) - window + 1)
    ]
