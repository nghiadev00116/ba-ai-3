import os
# Fix for Mac M1/M2/M3 Segmentation Faults with HuggingFace Tokenizers and PyTorch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import argparse
import faiss
import numpy as np
import time
from pypdf import PdfReader
try:
    import docx
except Exception:
    docx = None
from FlagEmbedding import BGEM3FlagModel


def normalize_embeddings(embs):
    """Normalize embeddings to unit length for cosine similarity."""
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.clip(norms, 1e-12, None)


def chunk_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def extract_text_from_docx(path):
    if docx is None:
        return None
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paragraphs)


def load_index_history(history_path):
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_index_history(history, history_path):
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def load_existing_index_and_chunks(index_path, chunks_path, dim):
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        print(f"Loading existing FAISS index from {index_path}...")
        index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        return index, chunks_data
    
    print("No existing FAISS index found. Creating a new one...")
    return faiss.IndexFlatIP(dim), []


def build_index(pdf_dir, index_path, chunks_path, history_path, model_name, batch_size, chunk_size, files_per_batch=10):
    print(f"Loading embedding model {model_name}...")
    model = BGEM3FlagModel(model_name, use_fp16=False)
    dim = 1024 # BAAI/bge-m3 default dimension
    
    index, all_chunks_data = load_existing_index_and_chunks(index_path, chunks_path, dim)
    index_history = load_index_history(history_path)

    files_to_process = []
    
    # Scan directory for new or modified files
    for fname in sorted(os.listdir(pdf_dir)):
        path = os.path.join(pdf_dir, fname)
        if not (fname.lower().endswith(".pdf") or fname.lower().endswith(".docx")):
            continue
            
        mtime = os.path.getmtime(path)
        
        # Check if file has been processed and hasn't changed
        if fname in index_history and index_history[fname] == mtime:
            continue
            
        files_to_process.append((fname, path, mtime))

    if not files_to_process:
        print("No new or modified documents found in directory:", pdf_dir)
        return

    print(f"Found {len(files_to_process)} new/modified files to index.")
    
    # Process files in batches
    for i in range(0, len(files_to_process), files_per_batch):
        batch_files = files_to_process[i:i+files_per_batch]
        print(f"\nProcessing batch {i//files_per_batch + 1} ({len(batch_files)} files)...")
        
        batch_chunks = []
        batch_metadata = []
        processed_in_batch = {}
        
        for fname, path, mtime in batch_files:
            print(f"  Reading {fname}...")
            try:
                if fname.lower().endswith(".pdf"):
                    reader = PdfReader(path)
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            chunks = chunk_text(text, chunk_size=chunk_size)
                            for chunk in chunks:
                                batch_chunks.append(chunk)
                                batch_metadata.append({"file": fname, "page": page_num})
                elif fname.lower().endswith(".docx"):
                    text = extract_text_from_docx(path)
                    if text:
                        chunks = chunk_text(text, chunk_size=chunk_size)
                        for page_num, chunk in enumerate(chunks):
                            batch_chunks.append(chunk)
                            batch_metadata.append({"file": fname, "page": page_num})
                            
                processed_in_batch[fname] = mtime
            except Exception as e:
                print(f"  Error reading {fname}: {e}")
        
        if batch_chunks:
            print(f"  Encoding {len(batch_chunks)} chunks from this batch...")
            embs = model.encode(batch_chunks, batch_size=batch_size, max_length=512)['dense_vecs']
            embs = np.array(embs).astype("float32")
            
            # Normalize for cosine similarity
            print("  Normalizing embeddings...")
            embs = normalize_embeddings(embs)
            
            # Append to FAISS index
            print("  Adding to FAISS index...")
            index.add(embs)
            
            # Append to chunks JSON data
            for t, m in zip(batch_chunks, batch_metadata):
                all_chunks_data.append({"text": t, "file": m["file"], "page": m["page"]})
                
        # Update history
        index_history.update(processed_in_batch)
        
        # Save state to disk immediately after each batch
        print("  Saving current batch progress to disk...")
        faiss.write_index(index, index_path)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks_data, f, ensure_ascii=False)
        save_index_history(index_history, history_path)
        
        print("  Batch complete. Memory freed.")

    print("\nAll files processed successfully!")
    print("FAISS Index saved:", index_path)
    print("Chunks JSON saved:", chunks_path)
    print("History log saved:", history_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from PDFs/DOCX in a folder")
    parser.add_argument("--docs", default="pdf_docs", help="Directory containing PDFs/DOCX")
    parser.add_argument("--index", default="faiss.index", help="Path to write FAISS index")
    parser.add_argument("--chunks", default="chunks.json", help="Path to write chunks+metadata (json)")
    parser.add_argument("--history", default="index_history.json", help="Path to write indexing history")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Embedding model name")
    parser.add_argument("--batch", type=int, default=8, help="Embedding batch size")
    parser.add_argument("--chunk_size", type=int, default=200, help="Words per chunk")
    parser.add_argument("--files_per_batch", type=int, default=10, help="Number of files to process per batch before saving to disk")

    args = parser.parse_args()

    build_index(
        pdf_dir=args.docs,
        index_path=args.index,
        chunks_path=args.chunks,
        history_path=args.history,
        model_name=args.model,
        batch_size=args.batch,
        chunk_size=args.chunk_size,
        files_per_batch=args.files_per_batch
    )
