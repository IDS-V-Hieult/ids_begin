# Giải Thích Chi Tiết Kiến Trúc Transformer

## Mục Lục
- [1. Tổng Quan](#1-tổng-quan)
- [2. Tokenization và Xử Lý Đầu Vào](#2-tokenization-và-xử-lý-đầu-vào)
- [3. Special Tokens và Position Encodings](#3-special-tokens-và-position-encodings)
- [4. Kiến Trúc Chi Tiết](#4-kiến-trúc-chi-tiết)
- [5. Quá Trình Training](#5-quá-trình-training)
- [6. Ứng Dụng và Tối Ưu](#6-ứng-dụng-và-tối-ưu)

## 1. Tổng Quan

### 1.1. Giới Thiệu
Transformer là kiến trúc mạng neural được thiết kế đặc biệt cho xử lý ngôn ngữ tự nhiên (NLP), được giới thiệu trong paper "Attention Is All You Need" (2017).

### 1.2. Đặc Điểm Nổi Bật
| Đặc điểm | Giải thích | Lợi ích |
|----------|------------|----------|
| Parallel Processing | Xử lý đồng thời nhiều tokens | Tăng tốc độ training và inference |
| Self-Attention | Tập trung vào các phần quan trọng của input | Hiểu ngữ cảnh tốt hơn |
| Position-aware | Nhận biết vị trí của từng token | Giữ được thông tin về thứ tự từ |

## 2. Tokenization và Xử Lý Đầu Vào

### 2.1. Quá Trình Tokenization

```python
# Ví dụ về tokenization
text = "Transformer là mô hình xử lý ngôn ngữ"
tokens = tokenizer.tokenize(text)
# Kết quả: ['Trans', '##former', 'là', 'mô', 'hình', 'xử', 'lý', 'ngôn', 'ngữ']
```

#### Các Phương Pháp Tokenization:

1. **Word-based Tokenization**:
   - Ưu điểm: Dễ hiểu, trực quan
   - Nhược điểm: Từ điển lớn, không xử lý được từ mới
   ```python
   # Word-based
   "Tôi yêu Việt Nam" -> ["Tôi", "yêu", "Việt_Nam"]
   ```

2. **BPE (Byte Pair Encoding)**:
   - Ưu điểm: Xử lý được từ mới, từ điển nhỏ
   - Cách hoạt động: Tách từ thành các cặp ký tự phổ biến
   ```python
   # BPE
   "transformer" -> ["trans", "former"]
   ```

3. **WordPiece**:
   - Kết hợp word-based và BPE
   - Sử dụng '##' để đánh dấu subword
   ```python
   # WordPiece
   "transformer" -> ["trans", "##former"]
   ```

## 3. Special Tokens và Position Encodings

### 3.1. Special Tokens

| Token | Chức năng | Ứng dụng |
|-------|-----------|-----------|
| [CLS] | - Đánh dấu bắt đầu câu<br>- Chứa thông tin tổng hợp của cả câu | - Classification tasks<br>- Next sentence prediction |
| [SEP] | - Phân tách các câu<br>- Đánh dấu kết thúc câu | - Sentence pair tasks<br>- Question answering |
| [MASK] | - Đánh dấu từ bị che trong MLM<br>- Dùng trong pre-training | - Masked language modeling<br>- BERT pre-training |
| [PAD] | - Đệm câu ngắn thành độ dài cố định | - Batch processing<br>- Tối ưu tính toán |

```python
# Ví dụ sử dụng special tokens
text = "Tôi yêu Việt Nam"
tokens = ["[CLS]", "Tôi", "yêu", "Việt", "Nam", "[SEP]"]

# Trong BERT sentence pair
text_pair = ["[CLS]", "Câu_1", "[SEP]", "Câu_2", "[SEP]"]
```

### 3.2. Position Encodings

#### Tại sao cần Position Encodings?
- Transformer xử lý song song → mất thông tin về thứ tự
- Position Encodings giúp mô hình biết vị trí của từng token

#### Các loại Position Encodings:

1. **Sinusoidal Position Encodings**:
```python
def get_sinusoid_encoding(pos, d_model):
    PE = np.zeros((pos, d_model))
    for i in range(pos):
        for j in range(0, d_model, 2):
            PE[i, j] = np.sin(i / (10000 ** (j/d_model)))
            PE[i, j+1] = np.cos(i / (10000 ** (j/d_model)))
    return PE
```

Ưu điểm:
- Không cần học
- Có thể xử lý câu dài hơn lúc training
- Có tính chu kỳ và liên tục

2. **Learned Position Embeddings**:
```python
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model))
```

Ưu điểm:
- Linh hoạt hơn
- Có thể học patterns phức tạp
- Tối ưu cho dataset cụ thể

## 4. Kiến Trúc Chi Tiết

### 4.1. Multi-Head Attention

Multi-Head Attention cho phép mô hình học nhiều representation khác nhau:

```python
def multi_head_attention(query, key, value, num_heads=8):
    # Chia thành các heads
    batch_size = query.size(0)
    head_dim = d_model // num_heads
    
    # Reshape và transform
    q = query.view(batch_size, -1, num_heads, head_dim)
    k = key.view(batch_size, -1, num_heads, head_dim)
    v = value.view(batch_size, -1, num_heads, head_dim)
    
    # Tính attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    attention = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attention, v)
    
    return output.reshape(batch_size, -1, d_model)
```

#### Giải thích chi tiết:

1. **Query, Key, Value Transformations**:
   - Query: "Tôi đang tìm gì?"
   - Key: "Những thông tin nào có sẵn?"
   - Value: "Thông tin thực tế là gì?"

2. **Scale Dot-Product Attention**:
   ```
   Attention(Q,K,V) = softmax(QK^T/√d_k)V
   ```
   - Chia cho √d_k để tránh gradients quá lớn
   - Softmax để normalize scores

3. **Multiple Heads**:
   - Mỗi head học một khía cạnh khác nhau
   - Head 1: Có thể tập trung vào ngữ pháp
   - Head 2: Có thể tập trung vào ngữ nghĩa
   - v.v...

## 5. Quá Trình Training

### 5.1. Adam Optimizer

Adam (Adaptive Moment Estimation) là optimizer kết hợp ưu điểm của:
- RMSprop: Adaptive learning rates
- Momentum: Tích lũy gradient

```python
# Pseudo-code của Adam
def adam_update(params, grads, m, v, t, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    # m: first moment (momentum)
    # v: second moment (RMSprop)
    
    # Update biased first moment
    m = b1 * m + (1 - b1) * grads
    
    # Update biased second moment
    v = b2 * v + (1 - b2) * grads**2
    
    # Correct bias
    m_hat = m / (1 - b1**t)
    v_hat = v / (1 - b2**t)
    
    # Update parameters
    params = params - lr * m_hat / (sqrt(v_hat) + eps)
```

#### Ưu điểm của Adam:
1. **Adaptive Learning Rates**:
   - Mỗi parameter có learning rate riêng
   - Tự động điều chỉnh theo gradient history

2. **Momentum**:
   - Giúp vượt qua local minima
   - Tăng tốc convergence

3. **RMSprop Component**:
   - Normalize gradients
   - Tránh overshooting

### 5.2. Learning Rate Scheduling

```python
def warmup_cosine_decay(step, warmup_steps, total_steps):
    if step < warmup_steps:
        # Linear warmup
        return float(step) / float(max(1, warmup_steps))
    else:
        # Cosine decay
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
```

#### Tại sao cần Learning Rate Scheduling?
1. **Warmup Phase**:
   - Bắt đầu với lr nhỏ
   - Tăng dần đến giá trị tối ưu
   - Tránh unstable gradients lúc đầu

2. **Decay Phase**:
   - Giảm dần lr theo thời gian
   - Fine-tune để tìm minimum tốt hơn
   - Tránh overshooting ở cuối training

## 6. Ứng Dụng và Tối Ưu

### 6.1. Memory Optimization

```python
# Gradient Checkpointing
class TransformerWithCheckpointing(nn.Module):
    def forward(self, x):
        # Không lưu activations của tất cả layers
        # Chỉ lưu một số checkpoint và tính lại khi cần
        return checkpoint_sequential(self.layers, 3, x)
```

### 6.2. Training Optimization

1. **Mixed Precision Training**:
```python
with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
```

2. **Gradient Accumulation**:
```python
for i, batch in enumerate(dataloader):
    output = model(batch)
    loss = criterion(output, target)
    (loss / accumulation_steps).backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 6.3. Inference Optimization

1. **Beam Search**:
```python
def beam_search(model, input, beam_width=5):
    # Giữ track của top-k sequences
    sequences = [([], 0)]  # (sequence, score)
    
    for _ in range(max_len):
        candidates = []
        for seq, score in sequences:
            # Predict next tokens
            logits = model(seq)
            probs = F.softmax(logits, dim=-1)
            
            # Get top-k next tokens
            top_k = torch.topk(probs, k=beam_width)
            
            for token, prob in zip(top_k.indices, top_k.values):
                candidates.append((seq + [token], score + math.log(prob)))
        
        # Keep top-k sequences
        sequences = sorted(candidates, key=lambda x: x[1])[-beam_width:]
```

## Kết Luận và Best Practices

### Performance Tips:
1. **Training**:
   - Sử dụng gradient checkpointing cho mô hình lớn
   - Mixed precision training (FP16)
   - Gradient accumulation cho batch size lớn

2. **Inference**:
   - Beam search cho generation tasks
   - Caching key-value pairs
   - Batch processing

### Debugging Tips:
1. **Loss không giảm**:
   - Kiểm tra learning rate
   - Thử warmup schedule
   - Verify gradient flow

2. **Memory issues**:
   - Gradient checkpointing
   - Reduce model size
   - Mixed precision training

---

*Lưu ý: Tài liệu này cung cấp hiểu biết sâu về Transformer. Để triển khai, cần thêm nhiều testing và tuning cho use case cụ thể.*