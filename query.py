import json
import argparse
from FlagEmbedding import BGEM3FlagModel
import faiss
import numpy as np


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

        D, I = index.search(q_emb, k=topk)

        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            if dist >= threshold:
                continue
            item = data[idx]
            results.append({"text": item.get("text"), "file": item.get("file"), "page": item.get("page"), "score": float(dist)})

        if not results:
            print("No relevant passages (try increasing --topk or threshold).")
            continue

        for i, r in enumerate(results, 1):
            print(f"[{i}] ({r['file']} - page {r['page']}) score={r['score']:.4f}")
            snippet = r['text']
            print(snippet[:800].replace('\n', ' '))
            print("---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a persisted FAISS index")
    parser.add_argument("--index", default="faiss.index", help="FAISS index path")
    parser.add_argument("--chunks", default="chunks.json", help="Chunks+metadata JSON path")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Embedding model name")
    parser.add_argument("--topk", type=int, default=5, help="Number of candidates to retrieve")
    parser.add_argument("--threshold", type=float, default=0.6, help="Distance threshold to filter results")
    args = parser.parse_args()

    query_loop(args.index, args.chunks, args.model, args.topk, args.threshold)
