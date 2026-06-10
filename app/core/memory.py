import json

import redis

from app.config import settings

# single Redis connection pool shared across all requests
_r = redis.from_url(settings.redis_url, decode_responses=True)


def get_history(session_id: str, n: int = 10) -> list[dict]:
    """Fetch the last n conversation turns for a session from Redis.

    Args:
        session_id: Unique identifier for the chat session.
        n: Maximum number of recent turns to retrieve.

    Returns:
        List of dicts with keys 'role' and 'content'.
    """
    raw: list[str] = _r.lrange(f"session:{session_id}", -n, -1)
    return [json.loads(m) for m in raw]


def save_turn(session_id: str, role: str, content: str) -> None:
    """Append a single turn to the session history in Redis.

    Args:
        session_id: Unique identifier for the chat session.
        role: Speaker role — 'user' or 'assistant'.
        content: Message text for this turn.
    """
    _r.rpush(f"session:{session_id}", json.dumps({"role": role, "content": content}))
    _r.expire(f"session:{session_id}", 86400)  # auto-expire after 24 hours
