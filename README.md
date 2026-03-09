# Dự án: RAG Chatbot AI

Dự án này là một hệ thống Hỏi-Đáp (QA) tự động dựa trên tài liệu PDF/DOCX, kết hợp giữa mô hình tìm kiếm vector **FAISS** và mô hình sinh ngôn ngữ lớn **Google Gemini**. Hệ thống có khả năng **nhớ lịch sử trò chuyện** của từng người dùng để mang lại trải nghiệm tự nhiên.

## Tính năng chính

- **Xử lý tài liệu:** Trích xuất văn bản từ file PDF/DOCX, chia nhỏ (chunking), và tạo vector nhúng (embedding) bằng mô hình `BAAI/bge-m3`.
- **Tìm kiếm cục bộ (Vector Search):** Tốc độ cực nhanh nhờ FAISS (Cosine Similarity).
- **Hỏi đáp thông minh:** Sử dụng API của Google Gemini (Flash) để phân tích tài liệu tìm được và trả lời đúng trọng tâm.
- **Lịch sử trò chuyện:**
  - Ghi nhớ ngữ cảnh đối thoại theo từng `session_id`.
  - Tự động xóa lịch sử nếu user không tương tác quá thời gian quy định (10 phút).
- **Cấu hình động:** Toàn bộ tinh chỉnh AI và máy chủ được đưa ra file `.env` tiện lợi.

---

## 🚀 Cài đặt môi trường

Dự án yêu cầu **Python 3.1x**.

1. **Tạo môi trường ảo (Khuyến nghị):**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. **Cài đặt thư viện:**
   Dự án đã có danh sách các phiên bản thư viện đang hoạt động ổn định nhất:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Cấu hình (File `.env`)

Hệ thống hoạt động dựa trên cấu hình môi trường. Bạn **phải** tạo một file `.env` ở thư mục gốc của dự án. Xem ví dụ mẫu:

```env
# --- GEMINI (LLM) ---
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_MAX_TOKENS=800000

# --- LỊCH SỬ CHAT ---
# Số tin nhắn nhớ được cho mỗi người dùng (20 tin = 10 vòng hỏi đáp)
MAX_HISTORY_MESSAGES=20
SESSION_EXPIRY_MINUTES=10

# --- CẤU HÌNH SERVER ---
SERVER_HOST=127.0.0.1
SERVER_PORT=8000

# --- TÌM KIẾM VECTOR ---
EMBEDDING_MODEL=BAAI/bge-m3
CHUNK_SIZE=200
DEFAULT_TOP_K=20
DEFAULT_THRESHOLD=0.35
```

---

## 🛠 Cách sử dụng (Luồng hoạt động)

### Bước 0: Tạo dữ liệu ảo (Tùy chọn)

Nếu bạn chưa có sẵn các tài liệu PDF/DOCX để test, dự án có kèm theo các script sinh dữ liệu giả lập tự động vào thư mục `pdf_docs/`. Chạy **một trong các lệnh** dưới đây từ thư mục gốc của dự án:

```bash
python scripts/generate_mock_appliances.py      # Sinh 100 file DOCX (Điện máy)
python scripts/generate_mock_appliances_pdf.py  # Sinh 100 file PDF (Điện máy - có font tiếng Việt)
python scripts/generate_mock_docx.py            # Sinh 100 file DOCX (Điện thoại, Laptop)
```

### Bước 1: Nạp tài liệu (Build Index)

Đặt các file tài liệu (`.pdf`, `.docx`) của bạn vào thư mục `pdf_docs/`. Chạy lệnh sau để hệ thống bắt đầu "học" bài:

```bash
python build_index.py --docs pdf_docs --index faiss.index --chunks chunks.json
```

_Ghi chú: Nếu file cũ không thay đổi, lệnh này sẽ bỏ qua không quét lại nhờ có tracking file `index_history.json`._

### Bước 2: Khởi động Server API

Bạn có thể chạy server bằng script python thông thường:

```bash
python main.py
```

Hoặc chạy thông qua uvicorn (Khuyến nghị dùng để theo dõi log và tốc độ truy xuất tốt hơn):

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

_Server mặc định lắng nghe ở `http://127.0.0.1:8000` (Thay đổi trong file `.env`)_

---

## 🌐 API Documentation

Bạn có thể dùng Postman để test các Endpoint sau:

### 1. `POST /query` (Tham số Chat)

Để AI ghi nhớ lịch sử hội thoại, bạn cần truyền theo một `session_id` riêng biệt cho mỗi người dùng (hoặc mỗi phiên đăng nhập).

**Body (JSON):**

```json
{
  "session_id": "khachhang_abc123",
  "question": "Samsung A36 giá bao nhiêu?",
  "topk": 20,
  "threshold": 0.35
}
```

**\*Gợi ý:** Nếu bạn hỏi tiếp "Nó có chống nước không?" với cùng `session_id` đó, hệ thống sẽ tự hiểu "Nó" là Samsung A36.\*

### 2. `POST /rebuild` (Làm mới nền tảng dữ liệu)

Gọi API này nếu bạn mới ném thêm file PDF/DOCX vào thư mục máy chủ và muốn Web AI lập tức cập nhật tài liệu mới mà không cần tắt/bật lại server.

**Body (JSON):**

```json
{
  "docs_dir": "pdf_docs",
  "index_path": "faiss.index",
  "chunks_path": "chunks.json"
}
```

---

## 🧪 Công cụ Test bổ sung

- `python query.py`: Giao diện dòng lệnh (CLI Terminal) để test trò chuyện chay thay vì gọi API. (Cần chạy `build_index` trước).
- `python test_similarity.py`: In ra Top các tài liệu giống nhất với từ khoá để kiểm tra thuật toán Vector FAISS.
