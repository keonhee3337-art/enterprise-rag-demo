"""
main.py — FastAPI backend for the Kearney RAG Demo.
Run: uvicorn demo.backend.main:app --reload --port 8000
"""
import os
import uuid
import sys

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag import search, generate_answer
from db import save_turn, get_history, get_session_summary

app = FastAPI(title="Enterprise RAG Demo", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    session_id: str = ""


class SourceChunk(BaseModel):
    text: str
    source: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    session_id: str
    turn_count: int


FRONTEND = Path(__file__).parent.parent / "frontend" / "index.html"


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND)


@app.get("/health")
def health():
    return {"status": "ok", "service": "Kearney RAG Demo"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    session_id = req.session_id or str(uuid.uuid4())

    # 1. Retrieve relevant chunks from Pinecone
    chunks = search(req.query, top_k=4)

    # 2. Load conversation history for context continuity
    history = get_history(session_id)

    # 3. Generate grounded answer via GPT-4o
    answer = generate_answer(req.query, chunks, history)

    # 4. Persist this turn to Supabase
    save_turn(session_id, req.query, answer, chunks)

    summary = get_session_summary(session_id)

    return ChatResponse(
        answer=answer,
        sources=[SourceChunk(**c) for c in chunks],
        session_id=session_id,
        turn_count=summary["turn_count"],
    )


@app.get("/history/{session_id}")
def history(session_id: str):
    return get_session_summary(session_id)
