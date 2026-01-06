import os
import json
import argparse
from pypdf import PdfReader
try:
    import docx
except Exception:
    docx = None
from FlagEmbedding import BGEM3FlagModel
import faiss
import numpy as np


def chunk_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def extract_text_from_docx(path):
    if docx is None:
        return None
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paragraphs)


def build_index(pdf_dir, index_path, chunks_path, model_name, batch_size, chunk_size):
    model = BGEM3FlagModel(model_name, use_fp16=True)

    all_chunks = []
    metadata = []

    for fname in sorted(os.listdir(pdf_dir)):
        path = os.path.join(pdf_dir, fname)
        if fname.lower().endswith(".pdf"):
            reader = PdfReader(path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    chunks = chunk_text(text, chunk_size=chunk_size)
                    for chunk in chunks:
                        all_chunks.append(chunk)
                        metadata.append({"file": fname, "page": page_num})
        elif fname.lower().endswith(".docx"):
            text = extract_text_from_docx(path)
            if text:
                chunks = chunk_text(text, chunk_size=chunk_size)
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    metadata.append({"file": fname, "page": i})

    if not all_chunks:
        print("No document text found in directory:", pdf_dir)
        return

    print(f"Encoding {len(all_chunks)} chunks...")
    embs = model.encode(all_chunks, batch_size=batch_size, max_length=512)['dense_vecs']
    embs = np.array(embs).astype("float32")

    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)

    faiss.write_index(index, index_path)

    # save chunks + metadata together
    data = [
        {"text": t, "file": m["file"], "page": m["page"]}
        for t, m in zip(all_chunks, metadata)
    ]
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print("Index built and saved:", index_path)
    print("Chunks saved:", chunks_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from PDFs/DOCX in a folder")
    parser.add_argument("--docs", default="pdf_docs", help="Directory containing PDFs/DOCX")
    parser.add_argument("--index", default="faiss.index", help="Path to write FAISS index")
    parser.add_argument("--chunks", default="chunks.json", help="Path to write chunks+metadata (json)")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Embedding model name")
    parser.add_argument("--batch", type=int, default=8, help="Embedding batch size")
    parser.add_argument("--chunk_size", type=int, default=200, help="Words per chunk")

    args = parser.parse_args()

    build_index(args.docs, args.index, args.chunks, args.model, args.batch, args.chunk_size)
