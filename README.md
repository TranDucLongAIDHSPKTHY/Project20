# Hệ thống gợi ý phim đơn giản
Sử dụng lọc cộng tác: KNN với 3 độ đo similarity là cosine , msd, pearson.
## Hướng dẫn chạy:
### 1. Cài đặt Python 3.10.0
Bước 1: Tải Python tại: https://www.python.org/downloads/release/python-3100/
Bước 2: Kéo xuống phần Files, chọn Windows installer (64-bit), tải về và mở để cài đặt.
Lưu ý: tích chọn Add Python to PATH khi cài đặt.

### 2. Cài đặt các thư viện cần thiết
Bước 1: Mở terminal và chạy lệnh:
Copy lệnh: pip install pandas numpy matplotlib seaborn scikit-surprise streamlit
Bước 2: Nếu báo lỗi C++ : tải https://visualstudio.microsoft.com/visual-cpp-build-tools/ và cài hết 4 mục đầu sau đó Install. Khởi động lại máy và tiếp tục lại bước 2.

### 3. Chạy code chính
Bước 1: Mở file code/RS_CollaborativeFiltering.ipynb và chạy. Thời gian chạy khoảng 10-15 phút.

### 4. Chạy demo giao diện
Bước 1: Mở terminal, chạy lệnh:
Copy lệnh: streamlit run demo/RS_demo.py
Lưu ý: Lần đầu sẽ yêu cầu nhập Gmail vào terminal, nhập xong là được. Những lần sau, chỉ cần chạy lệnh trên là đủ.
