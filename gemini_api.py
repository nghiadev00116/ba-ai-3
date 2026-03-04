from google import genai
import json

import tiktoken

# Khởi tạo client chung trên cùng để tái sử dụng
_client = genai.Client(api_key="AIzaSyB4HwF-flp11ollq4FD5vseAx3QWp0UF3I")

def refine_query(raw_question, chat_history=None):
    """Sử dụng Gemini để tiền xử lý: sửa lỗi chính tả và tối ưu hóa từ khóa tìm kiếm, dựa trên ngữ cảnh lịch sử."""
    
    # Format chat history to string if available
    history_str = ""
    if chat_history:
        history_lines = []
        for msg in chat_history:
            role = "Người dùng" if msg["role"] == "user" else "Hệ thống"
            history_lines.append(f"{role}: {msg['text']}")
        history_str = "Lịch sử trò chuyện gần đây:\n" + "\n".join(history_lines) + "\n\n"
        
    system_prompt = (
        "Bạn là một chuyên gia ngôn ngữ học và biên tập viên. Nhiệm vụ của bạn:\n"
        "1. Nhận một câu hỏi thô từ người dùng (có thể sai chính tả, lủng củng, dư thừa).\n"
        "2. Đọc 'Lịch sử trò chuyện gần đây' (nếu có) để hiểu ngữ cảnh. Nếu câu hỏi thô sử dụng các đại từ thay thế (như 'nó', 'vậy còn dòng kia', 'giá bao nhiêu'), hãy THAY THẾ đại từ đó bằng chủ ngữ cụ thể (ví dụ: tên sản phẩm) được nhắc đến trong lịch sử.\n"
        "3. Sửa lại các lỗi chính tả tiếng Việt nếu có.\n"
        "4. Lọc bỏ các từ ngữ giao tiếp dư thừa (như 'cho mình hỏi', 'ạ', 'dạ', 'ad ơi', 'xem giúp').\n"
        "5. Tóm gọn lại thành một câu hỏi đầy đủ ý nghĩa, giữ nguyên các từ khóa quan trọng (tên thương hiệu, thông số, bảo hành, v.v.) để hệ thống tìm kiếm trong cơ sở dữ liệu.\n"
        "6. Nếu câu hỏi thô đã có đủ thông tin và không phụ thuộc lịch sử, cứ giữ nguyên các từ khóa gốc của nó.\n"
        "7. TRẢ VỀ CHỈ MỘT DÒNG CÂU HỎI ĐÃ SỬA CHỮA, TRỌN VẸN NGỮ CẢNH, KHÔNG GIẢI THÍCH, KHÔNG CHÀO HỎI BIỂU CẢM GÌ THÊM."
    )
    
    prompt = f"{system_prompt}\n\n{history_str}Câu hỏi thô cần xử lý:\n{raw_question}"
    
    response = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    # Loại bỏ khoảng trắng hoặc ký tự thừa ở 2 đầu
    return response.text.strip()


def ask_gemini(payload):

    # Giới hạn của Gemini 1.5/2.0 Flash thường là 1 triệu tokens (input). 
    # Ta lấy 80% là khoảng 800,000 tokens. 
    # Tuy nhiên, để response nhanh và an toàn, ta có thể đặt limit thấp hơn tùy nhu cầu
    # Ở đây set cứng 800,000 tokens (80% của 1M)
    MAX_TOKENS = 800000 
    
    encoding = tiktoken.get_encoding("cl100k_base")
    
    valid_results = []
    current_tokens = 0
    
    # Tính toán token cho system prompt và user prompt tĩnh trước
    system_prompt = (
        "Bạn là một trợ lý phân tích dữ liệu chuyên nghiệp. Nhiệm vụ của bạn là:\n"
        "1. Nhận một câu hỏi (question) và một danh sách các đoạn trích dẫn (results).\n"
        "2. Tìm câu trả lời CHÍNH XÁC và NGẮN GỌN nhất dựa trên các văn bản đã nhận.\n"
        "3. KHÔNG ĐƯỢC tự bịa ra thông tin. Chỉ sử dụng thông tin có trong context.\n"
        "4. Nếu không tìm thấy thông tin phù hợp trong danh sách, hãy trả lời: \"Không tìm thấy câu trả lời trong tài liệu cung cấp.\"\n"
        "5. Nếu câu hỏi quá rộng hoặc không xác định được phạm vi (ví dụ: chỉ gửi từ 'samsung'), hãy yêu cầu người dùng làm rõ họ muốn biết thông tin gì cụ thể (bảo hành, giá cả, thông số kỹ thuật, v.v.).\n"
        "6. Tập trung vào vấn đề cốt lõi, không trả lời lan man.\n\n"
        "Định dạng trả về:\nHãy trả lời trực tiếp nội dung câu hỏi. Nếu có nhiều nguồn hỗ trợ cho câu trả lời, hãy tổng hợp chúng lại.\n"
    )
    
    base_prompt = f"{system_prompt}\nCÂU HỎI: {payload['question']}\n\nDANH SÁCH TRÍCH DẪN:\n\n\nHÃY PHÂN TÍCH VÀ TRẢ LỜI:\n"
    current_tokens += len(encoding.encode(base_prompt))
    
    for r in payload["results"]:
        chunk_text = f"- (file: {r['file']}, page: {r['page']})\n{r['text']}\n\n"
        chunk_tokens = len(encoding.encode(chunk_text))
        
        if current_tokens + chunk_tokens > MAX_TOKENS:
            print(f"Token limit reached! Truncating results. Tokens: {current_tokens}")
            break
            
        valid_results.append(r)
        current_tokens += chunk_tokens

    # Ghép context từ các đoạn trích dẫn ĐÃ ĐƯỢC KIỂM TRA LIMIT
    contextStrings = "\n\n".join([
        f"- (file: {r['file']}, page: {r['page']})\n{r['text']}" for r in valid_results
    ])

    # Thêm lịch sử trò chuyện vào prompt
    chat_history = payload.get("chat_history", [])
    history_str = ""
    if chat_history:
        history_lines = []
        for msg in chat_history:
            role = "Người dùng" if msg["role"] == "user" else "Hệ thống"
            history_lines.append(f"{role}: {msg['text']}")
        history_str = "LỊCH SỬ TRÒ CHUYỆN:\n" + "\n".join(history_lines) + "\n\n"

    # Prompt người dùng (user prompt)
    prompt = f"""
{system_prompt}
{history_str}DANH SÁCH TRÍCH DẪN TỪ TÀI LIỆU DƯỚI ĐÂY (dùng để trả lời câu hỏi hiện tại, ưu tiên hơn kiến thức chung):
{contextStrings}
\nCÂU HỎI HIỆN TẠI (Hãy hiểu theo ngữ cảnh lịch sử): {payload['question']}
\nHÃY PHÂN TÍCH VÀ TRẢ LỜI:
"""

    # Gọi model Gemini
    final_tokens = len(encoding.encode(prompt))

    print("="*50)
    print("PROMPT CHUẨN BỊ GỬI LÊN GEMINI:")
    print("="*50)
    print(prompt)
    print("="*50)
    print(f"-> TỔNG SỐ TOKEN SẼ GỬI: {final_tokens} tokens")
    print("="*50)

    response = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

# Cho phép chạy độc lập để test
if __name__ == "__main__":
    payload = {
        "question": "samsung A36 giá bao nhiêu",
        "results": [
            {
                "text": "DANH MỤC SẢN PHẨM SAMSUNG 2026 ... Galaxy A36 5G: Chống nước chuẩn IP67. Giá: 7.890.000 VNĐ. ... Bảo hành samsung 12 tháng chính hãng.",
                "file": "samsung2.pdf",
                "page": 0,
                "score": 0.5327
            },
        ]
    }
    print(ask_gemini(payload))
