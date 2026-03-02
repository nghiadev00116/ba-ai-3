# Dự án: BA-AI-3

Mô tả ngắn: repo này xử lý embedding từ tài liệu PDF và tìm các đoạn liên quan bằng FAISS.

Tệp chính:
- `main.py`: điểm chạy chính — đọc PDF từ thư mục `pdf_docs`, chia văn bản thành đoạn (chunks), sinh embedding bằng `BGEM3FlagModel`, tạo index FAISS và thực hiện truy vấn ví dụ.

Phụ thuộc chính (được suy ra từ `main.py`):
- `pypdf`
- `flagembedding` (gói hiển thị dưới dạng `FlagEmbedding` trong mã)
- `faiss` (hoặc `faiss-cpu` tùy nền tảng)
- `numpy`

Môi trường Python:
- Dự án có virtualenv: `bge_env/` (Python 3.13 theo cấu trúc thư mục). Sử dụng virtualenv này hoặc tạo một môi trường mới với Python 3.13.

Hướng dẫn nhanh — cục bộ (macOS / zsh):
1. Kích hoạt virtualenv có sẵn (nếu muốn dùng môi trường có sẵn):

```bash
source bge_env/bin/activate
```

2. Nếu bạn không dùng virtualenv có sẵn, tạo và kích hoạt một môi trường mới:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Cài phụ thuộc (nếu bạn có `requirements.txt`, dùng thay thế):

```bash
pip install pypdf flagembedding numpy faiss-cpu
```

Ghi chú: Trên một số hệ (GPU / nhiều nền tảng), tên gói `faiss` có thể khác (ví dụ `faiss-gpu`). Điều chỉnh cho phù hợp.

Chạy chương trình:

```bash
python main.py
```

Build index once and query many times:

- Tạo index (chỉ chạy 1 lần, lưu ra `faiss.index` và `chunks.json`):

```bash
python build_index.py --docs pdf_docs --index faiss.index --chunks chunks.json
```

- Mở chế độ tương tác để hỏi nhiều câu (không cần đọc lại file mỗi lần):

```bash
python query.py --index faiss.index --chunks chunks.json
```

Tham số hữu ích:
- `--topk`: số ứng viên trả về (mặc định 5)
- `--threshold`: ngưỡng cosine similarity (0–1, mặc định 0.7). Giá trị cao hơn = chặt chẽ hơn. Thử 0.65–0.8 tùy yêu cầu.

HTTP API (dùng Postman):

- Chạy server bằng `uvicorn` hoặc trực tiếp bằng `python main.py`:

```bash
# bằng uvicorn (khuyến nghị)
uvicorn main:app --host 127.0.0.1 --port 8000

# hoặc trực tiếp (chạy uvicorn nội bộ)
python main.py
```

- Endpoint hỏi:

POST http://127.0.0.1:8000/query

Body (JSON):

```json
{
	"question": "Câu hỏi của bạn",
	"topk": 5,
  "threshold": 0.7
}
```

**Ghi chú về threshold**: Sau khi chuyển sang cosine similarity, `threshold` là giá trị từ 0–1 (0.7 nghĩa là cosine >= 0.7). Giá trị cao hơn = kết quả khớp chặt chẽ hơn.

```json
{
	"docs_dir": "pdf_docs",
	"index_path": "faiss.index",
	"chunks_path": "chunks.json"
}
```

Ghi chú: server sẽ giữ index, chunks và mô hình trong bộ nhớ (cache) để trả lời nhiều yêu cầu mà không phải đọc lại file.

Thiết kế & ghi chú kỹ thuật:
- `main.py` dùng `BGEM3FlagModel('BAAI/bge-m3')` để sinh embedding.
- Văn bản từ PDF được chia theo số từ (`chunk_size=200`) rồi mã hóa theo lô (`batch_size=8`).
- **Cosine similarity**: Tất cả embeddings được chuẩn hoá (normalized) và FAISS index dùng `IndexFlatIP` (inner product trên vector chuẩn = cosine similarity).
- Kết quả trả về có `score` là cosine similarity (0–1, cao hơn = liên quan hơn).

Thay đổi gợi ý / bước tiếp theo:
- Thêm `requirements.txt` hoặc `pyproject.toml` để quản lý phụ thuộc rõ ràng.
- Viết script/CLI để cho phép nhập câu hỏi từ người dùng (hiện `query` trong `main.py` là cố định).
- Lưu index FAISS ra file để không phải rebuild mỗi lần.

Chạy thử query.py (giao diện dòng lệnh):

```bash
source bge_env/bin/activate
python query.py --index faiss.index --chunks chunks.json --topk 5 --threshold 0.6
```
Sau đó nhập câu hỏi trực tiếp, ví dụ: `samsung A36 giá bao nhiêu` và nhấn Enter để xem kết quả.

Chạy kiểm tra điểm tương đồng (test_similarity.py):

```bash
source bge_env/bin/activate
python test_similarity.py
```
Script này sẽ in ra top 10 đoạn văn có điểm cosine similarity cao nhất với câu hỏi mẫu, giúp bạn kiểm tra độ khớp thực tế giữa câu hỏi và dữ liệu.

Liên hệ/ tác giả: (Bạn có thể thêm thông tin tác giả hoặc liên hệ ở đây.)
