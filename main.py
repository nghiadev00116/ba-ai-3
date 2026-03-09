import os
import json
from typing import List, Dict, Any
from FlagEmbedding import BGEM3FlagModel
import faiss
import numpy as np
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from functools import lru_cache
from dotenv import load_dotenv

from gemini_api import ask_gemini, refine_query

load_dotenv()

app = FastAPI(title="Document QA API")

CHAT_HISTORY_FILE = "chat_history.json"
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", 20))  # Lấy N tin nhắn gần nhất
SESSION_EXPIRY_MINUTES = int(os.getenv("SESSION_EXPIRY_MINUTES", 10))
SESSION_EXPIRY_SECONDS = SESSION_EXPIRY_MINUTES * 60

def load_chat_history() -> Dict[str, Any]:
    """Load chat history and timestamps from JSON file."""
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Cleanup expired sessions upon load
                current_time = time.time()
                cleaned_data = {}
                for sid, info in data.items():
                    # Handle legacy format without timestamps or check expiry
                    if isinstance(info, dict) and "last_accessed" in info:
                        if current_time - info["last_accessed"] <= SESSION_EXPIRY_SECONDS:
                            cleaned_data[sid] = info
                    else:
                        # Legacy format (list directly) or invalid - reset
                        # We don't migrate legacy formats to avoid keeping stale data forever
                        pass
                return cleaned_data
        except Exception:
            return {}
    return {}

def save_chat_history(history: Dict[str, Any]):
    """Save chat history and timestamps to JSON file."""
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def normalize_embeddings(embs):
    """Normalize embeddings to unit length for cosine similarity."""
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.clip(norms, 1e-12, None)

INDEX_PATH = "faiss.index"
CHUNKS_PATH = "chunks.json"
DOCS_DIR = "pdf_docs"

# System configs
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", 8000))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 20))
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", 0.35))


class QueryRequest(BaseModel):
    session_id: str = "default"
    question: str
    topk: int = DEFAULT_TOP_K
    threshold: float = DEFAULT_THRESHOLD  # Cosine similarity threshold


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

    raw_q = req.question.strip()
    if not raw_q:
        raise HTTPException(status_code=400, detail="Empty question")

    # Load history
    current_time = time.time()
    all_histories = load_chat_history()
    
    session_data = all_histories.get(req.session_id, {"messages": [], "last_accessed": current_time})
    chat_history = session_data.get("messages", [])

    # Tiền xử lý câu hỏi qua Gemini để sửa lỗi và lọc từ khóa nhận biết ngữ cảnh
    print(f"\n[QUERY] Session: '{req.session_id}' | Câu hỏi gốc: '{raw_q}'")
    q = refine_query(raw_q, chat_history)
    print(f"[QUERY] Sau khi qua Gemini: '{q}'")

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
    payload = {
        "question": q,
        "results": results,
        "chat_history": chat_history
    }
    gemini_answer = ask_gemini(payload)

    # Save to history
    chat_history.append({"role": "user", "text": raw_q})
    chat_history.append({"role": "model", "text": gemini_answer})
    
    # Keep only recent messages
    if len(chat_history) > MAX_HISTORY_MESSAGES:
        chat_history = chat_history[-MAX_HISTORY_MESSAGES:]
        
    all_histories[req.session_id] = {
        "messages": chat_history,
        "last_accessed": time.time()
    }
    save_chat_history(all_histories)
    
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
    uvicorn.run("main:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
