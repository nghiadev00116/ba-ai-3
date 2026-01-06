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

Thiết kế & ghi chú kỹ thuật:
- `main.py` dùng `BGEM3FlagModel('BAAI/bge-m3')` để sinh embedding.
- Văn bản từ PDF được chia theo số từ (`chunk_size=200`) rồi mã hóa theo lô (`batch_size=8`).
- FAISS index sử dụng `IndexFlatL2` và ví dụ truy vấn trả về top-k kết quả.

Thay đổi gợi ý / bước tiếp theo:
- Thêm `requirements.txt` hoặc `pyproject.toml` để quản lý phụ thuộc rõ ràng.
- Viết script/CLI để cho phép nhập câu hỏi từ người dùng (hiện `query` trong `main.py` là cố định).
- Lưu index FAISS ra file để không phải rebuild mỗi lần.

Liên hệ/ tác giả: (Bạn có thể thêm thông tin tác giả hoặc liên hệ ở đây.)
