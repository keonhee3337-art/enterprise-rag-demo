"""
rag.py — Core RAG logic: embed query, search Pinecone, generate grounded answer.
"""
import os
from openai import OpenAI
from pinecone import Pinecone

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("kearney-demo")


def embed_query(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        input=text, model="text-embedding-3-small"
    )
    return response.data[0].embedding


def search(query: str, top_k: int = 4) -> list[dict]:
    """Embed query and retrieve top_k relevant chunks from Pinecone."""
    vector = embed_query(query)
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [
        {
            "text": m["metadata"]["text"],
            "source": m["metadata"]["source"],
            "score": round(m["score"], 3),
        }
        for m in results["matches"]
    ]


def generate_answer(query: str, chunks: list[dict], history: list[dict]) -> str:
    """Generate a grounded answer using retrieved chunks and conversation history."""
    context = "\n\n---\n\n".join(
        [f"[Source: {c['source']} | Relevance: {c['score']}]\n{c['text']}" for c in chunks]
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a consulting research assistant specializing in AI strategy. "
                "Answer questions concisely and professionally, grounded strictly in the provided context. "
                "Structure answers with bullet points where appropriate. "
                "If the context does not contain enough information, say so clearly rather than guessing. "
                "Always reference which source supports your key claims."
            ),
        }
    ]

    # Inject last 3 conversation turns for continuity
    for turn in history[-3:]:
        messages.append({"role": "user", "content": turn["query"]})
        messages.append({"role": "assistant", "content": turn["answer"]})

    messages.append({
        "role": "user",
        "content": f"Retrieved context:\n\n{context}\n\nQuestion: {query}",
    })

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=700,
    )
    return response.choices[0].message.content
