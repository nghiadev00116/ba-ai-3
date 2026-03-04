import os
import random
import urllib.request

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except ImportError:
    print("Vui lòng chạy lệnh: source bge_env/bin/activate && pip install reportlab")
    exit(1)

# Đưa output về thư mục gốc của dự án
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "pdf_docs")
NUM_FILES = 100

FONT_DIR = os.path.join(PROJECT_ROOT, "scripts", "fonts")
FONT_FILE = os.path.join(FONT_DIR, "Roboto-Regular.ttf")

# Tải font Roboto hỗ trợ tiếng Việt Unicode nếu chưa có
if not os.path.exists(FONT_DIR):
    os.makedirs(FONT_DIR)

if not os.path.exists(FONT_FILE):
    print("Đang tải font Roboto để hỗ trợ tiếng Việt trong PDF...")
    font_url = "https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Regular.ttf"
    urllib.request.urlretrieve(font_url, FONT_FILE)

# Đăng ký font vào reportlab
pdfmetrics.registerFont(TTFont('Roboto', FONT_FILE))

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
    lines.append(f"Thời gian bảo hành chung là {random.choice(['12', '24', '36'])} tháng. Riêng phần động cơ/máy nén (nếu có)")
    lines.append("thường được bảo hành mở rộng lên tới 10 năm.")
    lines.append("Liên hệ kĩ thuật viên hoặc tổng đài để yêu cầu thợ xuống hỗ trợ.")
    return lines

def create_pdf_vietnamese(filename, brand):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    textobject = c.beginText()
    textobject.setTextOrigin(50, height - 50)
    
    # Sử dụng font Roboto đã đăng ký
    textobject.setFont("Roboto", 12)
    
    content_lines = generate_random_content(brand)
    for line in content_lines:
        textobject.textLine(line)
        
    c.drawText(textobject)
    c.showPage()
    c.save()

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Đang bắt đầu tạo {NUM_FILES} file PDF Điện Máy giả tiếng Việt vào thư mục '{OUTPUT_DIR}'...")
    
    # Xóa các file docx rác lỡ tạo nhầm lúc nãy nếu người dùng muốn
    for f in os.listdir(OUTPUT_DIR):
        if "DienMay_TiengViet" in f and f.endswith(".docx"):
            os.remove(os.path.join(OUTPUT_DIR, f))
    
    for i in range(1, NUM_FILES + 1):
        brand = random.choice(brands)
        filename = os.path.join(OUTPUT_DIR, f"DienMay_TiengViet_{i:03d}_{brand.lower()}.pdf")
        create_pdf_vietnamese(filename, brand)
        
        if i % 20 == 0:
            print(f" Đã tạo {i}/{NUM_FILES} files PDF điện máy...")
            
    print("Hoàn tất! Cả đoạn chữ tiếng Việt đã được nhúng Font Unicode chuẩn nằm gọn trong file PDF.")
    print("Bạn có thể chạy lệnh 'python build_index.py' để nạp 100 file PDF Điện máy này vào FAISS Index.")

if __name__ == "__main__":
    main()
