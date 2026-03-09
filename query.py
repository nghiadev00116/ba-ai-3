import json
import argparse
from FlagEmbedding import BGEM3FlagModel
import faiss
import numpy as np


def normalize_embeddings(embs):
    """Normalize embeddings to unit length for cosine similarity."""
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.clip(norms, 1e-12, None)


def query_loop(index_path, chunks_path, model_name, topk, threshold):
    print("Loading index and chunks...")
    index = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = BGEM3FlagModel(model_name, use_fp16=True)

    print("Ready. Type a question (or 'exit' to quit).")
    while True:
        try:
            q = input("? ").strip()
        except EOFError:
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        q_emb = model.encode([q], max_length=256)['dense_vecs']
        q_emb = np.array(q_emb).astype("float32")
        # Normalize for cosine similarity
        q_emb = normalize_embeddings(q_emb)

        D, I = index.search(q_emb, k=topk)

        results = []
        for similarity, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            # With IndexFlatIP on normalized vectors, higher score = more similar
            if similarity < threshold:
                continue
            item = data[idx]
            results.append({"text": item.get("text"), "file": item.get("file"), "page": item.get("page"), "score": float(similarity)})

        if not results:
            print("No relevant passages (try increasing --topk or threshold).")
            continue

        for i, r in enumerate(results, 1):
            print(f"[{i}] ({r['file']} - page {r['page']}) similarity={r['score']:.4f}")
            snippet = r['text']
            print(snippet[:800].replace('\n', ' '))
            print("---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a persisted FAISS index")
    parser.add_argument("--index", default="faiss.index", help="FAISS index path")
    parser.add_argument("--chunks", default="chunks.json", help="Chunks+metadata JSON path")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Embedding model name")
    parser.add_argument("--topk", type=int, default=5, help="Number of candidates to retrieve")
    parser.add_argument("--threshold", type=float, default=0.6, help="Cosine similarity threshold (0-1, higher=more similar)")
    args = parser.parse_args()

    query_loop(args.index, args.chunks, args.model, args.topk, args.threshold)
