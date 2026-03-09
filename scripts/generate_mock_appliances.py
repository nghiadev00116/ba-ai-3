import os
import random
from docx import Document

# Đưa output về thư mục gốc của dự án kể cả khi chạy từ trong thư mục scripts/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "pdf_docs")
NUM_FILES = 100

brands = ["Samsung", "LG", "Sony", "Panasonic", "Toshiba", "Daikin", "Aqua", "Electrolux", "TCL", "Casper"]
products = ["Tivi (Smart TV)", "Tủ lạnh", "Máy giặt", "Máy lạnh (Điều hòa)", "Tủ đông", "Máy lọc không khí"]
features = [
    "Công nghệ Inverter tiết kiệm điện", "Màn hình 4K sắc nét", "Giặt sấy 2 trong 1", 
    "Khử mùi Nano Titanium", "Làm lạnh siêu tốc", "Hệ thống gió 4 chiều", 
    "Cảnh báo lỗi qua ứng dụng điện thoại", "Màn hình OLED 65 inch đỉnh cao", 
    "Chống rung, giảm ồn vượt trội", "Gas R32 thân thiện môi trường",
    "Bảo hành máy nén 10 năm", "Lọc bụi mịn PM2.5", "Tính năng tự động vệ sinh",
    "Bảo hành 24 tháng tận nhà", "Bảo hành 12 tháng chính hãng", "Hỗ trợ lắp đặt miễn phí"
]

def generate_random_content(brand):
    """Tạo ra danh sách đoạn văn bản tiếng Việt giả cho thiết bị điện máy"""
    lines = []
    lines.append(f"TÀI LIỆU HƯỚNG DẪN SỬ DỤNG VÀ THÔNG SỐ ĐIỆN MÁY: {brand.upper()}")
    lines.append("=" * 60)
    lines.append("")
    
    num_products = random.randint(3, 7)
    for i in range(num_products):
        prod_type = random.choice(products)
        series = f"{brand} {prod_type} - Đời mới {random.randint(2024, 2026)}"
        price = random.randint(3, 80) * 1000000
        
        lines.append(f"Danh mục {i+1}: {series}")
        lines.append(f"Giá bán thẻ niêm yết: {price:,} VNĐ")
        lines.append("Tính năng công nghệ:")
        for _ in range(random.randint(2, 5)):
            lines.append(f"  - {random.choice(features)}")
        lines.append("")
        
    lines.append("CHÍNH SÁCH BẢO HÀNH ĐIỆN MÁY:")
    lines.append(f"Tất cả sản phẩm điện máy gia dụng mang thương hiệu {brand} đều được hỗ trợ bảo hành ngay tại nhà.")
    lines.append(f"Thời gian bảo hành chung là {random.choice(['12', '24', '36'])} tháng. Riêng phần động cơ/máy nén (nếu có) thường được bảo hành mở rộng lên tới 10 năm.")
    lines.append("Liên hệ kĩ thuật viên hoặc tổng đài để yêu cầu thợ xuống hỗ trợ.")
    
    return lines

def create_docx(filename, brand):
    doc = Document()
    content_lines = generate_random_content(brand)
    
    for line in content_lines:
        doc.add_paragraph(line)
        
    doc.save(filename)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Đang bắt đầu tạo {NUM_FILES} file DOCX Điện Máy giả vào thư mục '{OUTPUT_DIR}'...")
    
    for i in range(1, NUM_FILES + 1):
        brand = random.choice(brands)
        filename = os.path.join(OUTPUT_DIR, f"DienMay_TiengViet_{i:03d}_{brand.lower()}.docx")
        create_docx(filename, brand)
        
        if i % 20 == 0:
            print(f" Đã tạo {i}/{NUM_FILES} files điện máy...")
            
    print("Hoàn tất! Bạn có thể chạy lệnh 'python build_index.py' để nạp 100 file Điện máy này vào FAISS Index.")

if __name__ == "__main__":
    main()
