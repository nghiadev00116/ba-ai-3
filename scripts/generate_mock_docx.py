import os
import random
from docx import Document

# Đưa output về thư mục gốc của dự án kể cả khi chạy từ trong thư mục scripts/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "pdf_docs")
NUM_FILES = 100

# Dữ liệu tiếng Việt để test
brands = ["Samsung", "Apple", "Xiaomi", "Oppo", "Sony", "LG", "Huawei", "Nokia", "Motorola", "Pixel"]
products = ["Điện thoại", "Máy tính bảng", "Tai nghe", "Đồng hồ thông minh", "Laptop", "Màn hình"]
features = [
    "Pin 5000mAh siêu trâu", "Pin 4000mAh", "Sạc siêu nhanh 120W", "Sạc nhanh 65W", 
    "Màn hình OLED 120Hz mượt mà", "Màn hình AMOLED hiện đại", 
    "Camera 200MP siêu nét", "Camera 108MP cảm biến lớn", "Hợp tác ống kính Leica", 
    "Chống nước và bụi bẩn chuẩn IP68", "Chống nước IP67",
    "Bảo hành 12 tháng tại các trung tâm", "Bảo hành 24 tháng chính hãng toàn quốc",
    "Chip Snapdragon 8 Gen 3 mạnh mẽ", "Chip Apple A17 Pro độc quyền", "Chip Dimensity 9300 mát mẻ"
]

def generate_random_content(brand):
    """Tạo ra danh sách đoạn văn bản tiếng Việt giả"""
    lines = []
    lines.append(f"TÀI LIỆU HƯỚNG DẪN VÀ THÔNG SỐ KỸ THUẬT SẢN PHẨM: {brand.upper()}")
    lines.append("=" * 50)
    lines.append("")
    
    num_products = random.randint(3, 7)
    for i in range(num_products):
        prod_type = random.choice(products)
        series = f"{brand} {prod_type} Series {random.randint(1, 20)}"
        price = random.randint(5, 40) * 1000000
        
        lines.append(f"Sản phẩm {i+1}: {series}")
        lines.append(f"Mức giá tham khảo tại thị trường Việt Nam: {price:,} VNĐ")
        lines.append("Đặc điểm nổi bật:")
        for _ in range(random.randint(2, 4)):
            lines.append(f"  - {random.choice(features)}")
        lines.append("")
        
    lines.append("CHÍNH SÁCH HỖ TRỢ VÀ BẢO HÀNH:")
    lines.append(f"Nếu thiết bị {brand} của bạn gặp sự cố phần cứng hoặc phần mềm, vui lòng mang đến trung tâm bảo hành gần nhất để được kiểm tra.")
    lines.append(f"Chính sách bảo hành tiêu chuẩn đang áp dụng cho toàn bộ dòng sản phẩm {brand} là {random.choice(['12', '18', '24'])} tháng kể từ ngày kích hoạt.")
    
    return lines

def create_docx(filename, brand):
    doc = Document()
    content_lines = generate_random_content(brand)
    
    for line in content_lines:
        # Nhập thẳng chuỗi tiếng Việt Unicode vào file Word
        doc.add_paragraph(line)
        
    doc.save(filename)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Đang bắt đầu tạo {NUM_FILES} file DOCX tiếng Việt giả vào thư mục '{OUTPUT_DIR}'...")
    
    for i in range(1, NUM_FILES + 1):
        brand = random.choice(brands)
        filename = os.path.join(OUTPUT_DIR, f"TaiLieu_TiengViet_{i:03d}_{brand.lower()}.docx")
        create_docx(filename, brand)
        
        if i % 20 == 0:
            print(f" Đã tạo {i}/{NUM_FILES} files...")
            
    print("Hoàn tất! Bạn có thể chạy lệnh 'python build_index.py' để test tính năng lập chỉ mục (Indexing).")

if __name__ == "__main__":
    main()
