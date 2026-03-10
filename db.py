"""
db.py — Conversation history via Supabase, with in-memory fallback.
"""
import os
from datetime import datetime, timezone

_memory: dict[str, list] = {}
_supabase = None


def _get_client():
    global _supabase
    if _supabase is not None:
        return _supabase
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_API_KEY", "")
    if url and key:
        try:
            from supabase import create_client
            _supabase = create_client(url, key)
            return _supabase
        except Exception as e:
            print(f"Supabase init failed: {e} — using in-memory fallback")
    return None


def save_turn(session_id: str, query: str, answer: str, sources: list):
    turn = {
        "session_id": session_id,
        "query": query,
        "answer": answer,
        "sources": [s["source"] for s in sources],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    client = _get_client()
    if client:
        try:
            client.table("conversations").insert(turn).execute()
            return
        except Exception as e:
            print(f"Supabase insert failed: {e} — falling back to memory")
    _memory.setdefault(session_id, []).append(turn)


def get_history(session_id: str, limit: int = 5) -> list:
    client = _get_client()
    if client:
        try:
            result = (
                client.table("conversations")
                .select("query,answer,created_at")
                .eq("session_id", session_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return list(reversed(result.data))
        except Exception as e:
            print(f"Supabase query failed: {e} — using memory")
    return _memory.get(session_id, [])[-limit:]


def get_session_summary(session_id: str) -> dict:
    history = get_history(session_id, limit=20)
    return {
        "session_id": session_id,
        "turn_count": len(history),
        "topics": [h["query"][:60] + "..." if len(h["query"]) > 60 else h["query"] for h in history],
    }
