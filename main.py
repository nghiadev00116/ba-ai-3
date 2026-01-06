import os
import json
from pypdf import PdfReader
from FlagEmbedding import BGEM3FlagModel
import faiss
import numpy as np

# Hàm chia nhỏ văn bản thành đoạn ngắn (chunk)
def chunk_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Khởi tạo model
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Thư mục chứa nhiều PDF
pdf_dir = "pdf_docs"

all_chunks = []
metadata = []

# Đọc tất cả PDF và chia nhỏ thành đoạn
for fname in os.listdir(pdf_dir):
    if fname.endswith(".pdf"):
        reader = PdfReader(os.path.join(pdf_dir, fname))
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                chunks = chunk_text(text, chunk_size=200)
                for chunk in chunks:
                    all_chunks.append(chunk)
                    metadata.append({"file": fname, "page": page_num})

# Sinh embedding cho toàn bộ chunk
embs = model.encode(all_chunks, batch_size=8, max_length=512)['dense_vecs']
embs = np.array(embs).astype("float32")

# Tạo FAISS index
index = faiss.IndexFlatL2(embs.shape[1])
index.add(embs)

# Câu hỏi
query = "thời gian bảo hành iphone bao lâu"
q_emb = model.encode([query], max_length=256)['dense_vecs']
q_emb = np.array(q_emb).astype("float32")

# Tìm top-5 đoạn liên quan
D, I = index.search(q_emb, k=5)

# Lọc kết quả theo ngưỡng khoảng cách và chỉ lấy top-3 ngắn nhất
threshold = 0.6
results = [
    {
        "text": all_chunks[idx],
        "file": metadata[idx]["file"],
        "page": metadata[idx]["page"],
        "score": float(dist)
    }
    for dist, idx in zip(D[0], I[0]) if dist < threshold
]
results = sorted(results, key=lambda x: len(x["text"]))[:3]

# Xuất JSON
output = {
    "question": query,
    "results": results
}

print(json.dumps(output, ensure_ascii=False, indent=2))
