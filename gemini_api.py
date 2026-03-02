from google import genai
import json

import tiktoken

def ask_gemini(payload):
    # Khởi tạo client với API key
    client = genai.Client(api_key="AIzaSyB4HwF-flp11ollq4FD5vseAx3QWp0UF3I")

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

    # Prompt người dùng (user prompt)
    prompt = f"""
{system_prompt}
CÂU HỎI: {payload['question']}
\nDANH SÁCH TRÍCH DẪN:
{contextStrings}
\nHÃY PHÂN TÍCH VÀ TRẢ LỜI:
"""

    # Gọi model Gemini
    response = client.models.generate_content(
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
