import os
import json
from typing import List
from FlagEmbedding import BGEM3FlagModel
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from functools import lru_cache
from gemini_api import ask_gemini

app = FastAPI(title="Document QA API")


def normalize_embeddings(embs):
    """Normalize embeddings to unit length for cosine similarity."""
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.clip(norms, 1e-12, None)

INDEX_PATH = "faiss.index"
CHUNKS_PATH = "chunks.json"
DOCS_DIR = "pdf_docs"
MODEL_NAME = "BAAI/bge-m3"


class QueryRequest(BaseModel):
    question: str
    topk: int = 20
    threshold: float = 0.6  # Cosine similarity threshold


class RebuildRequest(BaseModel):
    docs_dir: str = DOCS_DIR
    index_path: str = INDEX_PATH
    chunks_path: str = CHUNKS_PATH


def load_index_and_chunks(index_path=INDEX_PATH, chunks_path=CHUNKS_PATH):
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError("Index or chunks file not found. Build the index first.")
    index = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks


def ensure_model():
    return BGEM3FlagModel(MODEL_NAME, use_fp16=True)


# Simple in-memory cache for repeated identical queries
query_cache = {}


@app.on_event("startup")
def startup_event():
    global index, chunks, model
    try:
        index, chunks = load_index_and_chunks()
        print("Loaded existing index and chunks")
    except FileNotFoundError:
        index = None
        chunks = []
        print("No existing index found at startup")
    model = ensure_model()


@app.post("/query")
def query(req: QueryRequest):
    if index is None:
        raise HTTPException(status_code=400, detail="Index not available. Call /rebuild first.")

    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    cache_key = f"{q}|{req.topk}|{req.threshold}"
    if cache_key in query_cache:
        results = query_cache[cache_key]
    else:
        q_emb = model.encode([q], max_length=256)["dense_vecs"]
        q_emb = np.array(q_emb).astype("float32")
        q_emb = normalize_embeddings(q_emb)
        D, I = index.search(q_emb, k=req.topk)
        results = []
        for similarity, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            if similarity < req.threshold:
                continue
            item = chunks[idx]
            results.append({"text": item.get("text"), "file": item.get("file"), "page": item.get("page"), "score": float(similarity)})
        query_cache[cache_key] = results
    # Gọi Gemini
    payload = {"question": q, "results": results}
    gemini_answer = ask_gemini(payload)
    return {"gemini_answer": gemini_answer}


@app.post("/rebuild")
def rebuild(req: RebuildRequest):
    # import locally to avoid circular imports at module import time
    import build_index

    build_index.build_index(req.docs_dir, req.index_path, req.chunks_path, MODEL_NAME, batch_size=8, chunk_size=200)
    # reload
    global index, chunks
    index, chunks = load_index_and_chunks(req.index_path, req.chunks_path)
    query_cache.clear()
    return {"status": "ok", "index": req.index_path, "chunks": req.chunks_path}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
