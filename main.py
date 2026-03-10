"""
main.py — FastAPI backend for the Enterprise RAG Demo.
"""
import os
import uuid
from pathlib import Path

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


FRONTEND = Path(__file__).parent / "frontend" / "index.html"


@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND)


@app.get("/health")
def health():
    return {"status": "ok", "service": "Enterprise RAG Demo"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    session_id = req.session_id or str(uuid.uuid4())
    chunks = search(req.query, top_k=4)
    history = get_history(session_id)
    answer = generate_answer(req.query, chunks, history)
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
