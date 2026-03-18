# Local GPT Vision: Hệ thống RAG với khả năng xử lý hình ảnh hoàn toàn tại local

## 1. Giới thiệu

Local GPT Vision là một dự án mã nguồn mở mở rộng từ Local GPT, cho phép thực hiện Retrieval Augmented Generation (RAG) với khả năng xử lý hình ảnh. Khác với các hệ thống RAG truyền thống chỉ xử lý văn bản, Local GPT Vision có thể trích xuất và hiểu thông tin từ hình ảnh, bảng biểu và đồ thị trong tài liệu.

### 1.1. Tại sao cần Vision-based RAG?

Lấy ví dụ một báo cáo về biến đổi khí hậu:
- Chứa nhiều bảng biểu và đồ thị
- Thông tin quan trọng được nhúng trong hình ảnh
- Hệ thống RAG truyền thống sẽ bỏ sót thông tin này
- Vision-based RAG có thể hiểu và trích xuất toàn bộ thông tin

## 2. Kiến trúc Hệ thống

### 2.1. COPAL Architecture

```plaintext
Input Document -> PDF Pages -> Image Patches
                              |
Vision Encoder -> Embedding Space <- LLM Projections
                              |
User Query -> Retrieval -> Relevant Pages -> Generation
```

### 2.2. So sánh với RAG truyền thống

**RAG truyền thống:**
- Cần chia nhỏ văn bản
- Tính toán embeddings
- Phức tạp trong xử lý hình ảnh và bảng biểu
- Yêu cầu OCR và layout detection

**Vision-based RAG:**
- Chuyển tài liệu thành pages
- Sử dụng COPAL để tạo index
- Truy xuất dựa trên thông tin visual
- Đơn giản và hiệu quả hơn

## 3. Các Models Được Hỗ Trợ

### 3.1. Retrieval Models
- COPAL
- COQWEN2 (khuyến nghị)
- CLIP

### 3.2. Generation Models

**Open Source:**
- Llama Vision
- Pixol MMO
- QWEN Vision

**API Providers:**
- Gemini
- GPT-4V
- Grock (Llama API)

## 4. Hướng dẫn Cài đặt

### 4.1. Yêu cầu Hệ thống

```bash
# Tạo môi trường Conda
conda create -n vision-rag python=3.10
conda activate vision-rag

# Clone repository
git clone https://github.com/local-gpt-vision/local-gpt-vision
cd local-gpt-vision

# Cài đặt dependencies
pip install -r requirements.txt

# Cài đặt transformer dev version nếu cần
pip uninstall transformers
pip install git+https://github.com/huggingface/transformers
```

### 4.2. Cấu hình API Keys (nếu cần)

```plaintext
# .env file
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
GROCK_API_KEY=your_key
```

## 5. Sử dụng Thực tế

### 5.1. Ví dụ với Hóa đơn AWS

```plaintext
User: "Tổng chi phí trên hóa đơn AWS là bao nhiêu?"

# Sử dụng GPT-4
Response: "Tổng chi phí là $4.11"

# Sử dụng Gemini
Response: "Hóa đơn AWS có tổng chi phí là $4.11, 
          bao gồm chi tiết các khoản phí..."

# Sử dụng QWEN local
Response: "Chi phí khoảng 4 đến 4.11 đơn vị tiền tệ"
```

### 5.2. Xử lý Biểu đồ và Bảng

```plaintext
User: "Giải thích Figure 14"
Response: [Chi tiết giải thích về biểu đồ]

User: "Trích xuất Table 10 dưới dạng markdown"
Response: [Bảng dữ liệu được format dưới dạng markdown]
```

## 6. Điểm mạnh và Hạn chế

### 6.1. Điểm mạnh

1. **Xử lý Toàn diện:**
   - Hiểu được hình ảnh và bảng biểu
   - Không cần OCR
   - Tích hợp nhiều loại models

2. **Linh hoạt:**
   - Chạy hoàn toàn local
   - Hỗ trợ nhiều model providers
   - Dễ dàng chuyển đổi giữa các models

3. **User-friendly:**
   - Giao diện web đơn giản
   - Nhiều chat sessions
   - Dễ dàng upload tài liệu

### 6.2. Hạn chế

1. **Tài nguyên:**
   - Yêu cầu VRAM cao cho local models
   - Tốn memory khi switch sessions
   - Cần optimize hiệu năng

2. **Độ chính xác:**
   - Phụ thuộc vào chất lượng hình ảnh
   - Kết quả khác nhau giữa các models
   - Cần cải thiện system prompts

## 7. Hướng Phát triển

### 7.1. Cần cải thiện

- Tối ưu memory usage
- Thêm support cho nhiều định dạng file
- Cải thiện độ chính xác của trích xuất bảng
- Tích hợp thêm các models mới

### 7.2. Đóng góp

Dự án mong muốn nhận được đóng góp từ cộng đồng trong các lĩnh vực:
- Tích hợp models mới
- Cải thiện hiệu năng
- Thêm tính năng mới
- Sửa lỗi và tối ưu code

## Kết luận

Local GPT Vision là một bước tiến quan trọng trong lĩnh vực RAG, mang đến khả năng xử lý thông tin visual mà không cần thông qua các bước trung gian phức tạp. Dự án này đặc biệt hữu ích cho các ứng dụng cần trích xuất thông tin từ tài liệu có nhiều hình ảnh, bảng biểu và đồ thị.

*Bài viết được tổng hợp từ tài liệu về Local GPT Vision, một dự án mã nguồn mở được phát triển năm 2024.*