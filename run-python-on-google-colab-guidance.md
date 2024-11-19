# Hướng dẫn sử dụng Google Colab để chạy Python

## 1. Truy cập Google Colab
1. Mở trình duyệt web
2. Truy cập: [colab.research.google.com](https://colab.research.google.com)
3. Đăng nhập bằng tài khoản Google
4. Click "File" → "New Notebook" để tạo notebook mới

## 2. Thiết lập môi trường
### Kích hoạt GPU
1. Click "Runtime" → "Change runtime type"
2. Chọn "GPU" trong Hardware accelerator
3. Click "Save"

### Kiểm tra GPU
```python
# Kiểm tra GPU đã được kích hoạt
!nvidia-smi

### Menu chính:
- File: Quản lý notebook
- Edit: Chỉnh sửa
- View: Tùy chỉnh giao diện
- Insert: Thêm cell
- Runtime: Quản lý việc chạy code
- Tools: Công cụ bổ sung

# Kiểm tra phiên bản Python
!python --version

# Cài đặt thư viện
!pip install tên_thư_viện

# Kiểm tra GPU
!nvidia-smi

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

#Phím tắt hữu ích

Ctrl + Enter: Chạy cell hiện tại
Shift + Enter: Chạy cell và tạo cell mới
Ctrl + M B: Thêm cell phía dưới
Ctrl + M A: Thêm cell phía trên
Ctrl + M D: Xóa cell hiện tại

#Làm việc với Google Drive
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Truy cập files
PATH = '/content/drive/My Drive/your_folder/'


#Debug và Monitoring
# Kiểm tra memory
!free -h

# Kiểm tra GPU usage
!nvidia-smi

# Kiểm tra thời gian chạy
%%time
# code của bạn


#Tips và thủ thuật

.Sử dụng %%capture để ẩn output không cần thiết

. Dùng ! để chạy lệnh terminal

. Sử dụng %%writefile để tạo file

. Mount Google Drive để lưu trữ dữ liệu lâu dài