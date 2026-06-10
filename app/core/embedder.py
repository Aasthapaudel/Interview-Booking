from sentence_transformers import SentenceTransformer

# model loaded once at startup and reused across all requests
_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(texts: list[str]) -> list[list[float]]:
    """Generate 384-dimensional embeddings for a list of strings.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of float vectors, one per input string.
    """
    return _model.encode(texts, show_progress_bar=False).tolist()
