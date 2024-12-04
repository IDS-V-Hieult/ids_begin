# Google Transformer Architecture

## Table of Contents
- [1. Overview](#1-overview)
- [2. Main Components](#2-main-components)
- [3. Detailed Architecture](#3-detailed-architecture)
- [4. Training Process](#4-training-process)
- [5. Mathematical Foundations](#5-mathematical-foundations)
- [6. Implementation Tips](#6-implementation-tips)

## 1. Overview

Transformer là kiến trúc neural network được Google giới thiệu trong paper "Attention Is All You Need" (2017).

### Đặc điểm chính:
- Không sử dụng RNN hay CNN
- Hoàn toàn dựa trên attention mechanism
- Cho phép parallel processing
- Xử lý được long-range dependencies

## 2. Main Components

### 2.1. Encoder Stack
| Component | Function | Input/Output |
|-----------|----------|--------------|
| Self-Attention | Tìm mối quan hệ giữa tokens | Tokens → Attention Scores |
| Feed Forward | Xử lý sâu hơn | Attention Output → Features |
| Layer Norm | Chuẩn hóa | Features → Normalized Output |

### 2.2. Decoder Stack
| Component | Function | Input/Output |
|-----------|----------|--------------|
| Masked Attention | Tìm quan hệ trong output | Previous Tokens → Scores |
| Cross-Attention | Kết nối encoder-decoder | Encoder Output → Context |
| Feed Forward | Xử lý cuối cùng | Context → Final Output |

## 3. Detailed Architecture

### 3.1. Multi-Head Attention

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₙ)W
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

#### Components:
- **Query (Q)**: Vector tìm kiếm
- **Key (K)**: Vector chứa thông tin
- **Value (V)**: Thông tin thực tế
- **Heads**: Góc nhìn song song

### 3.2. Position Encoding

Các phương pháp:

1. **Sinusoidal**:
   ```python
   PE(pos,2i) = sin(pos/10000^(2i/d_model))
   PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
   ```

2. **Learned**:
   - Học trực tiếp từ data
   - Dạng embedding matrix

## 4. Training Process

### 4.1. Forward Pass
1. Tokenize input text
2. Add position encodings
3. Process through encoder:
   - Self-attention
   - Feed-forward
   - Layer normalization
4. Process through decoder:
   - Masked attention
   - Cross-attention
   - Feed-forward
5. Linear + Softmax

### 4.2. Hyperparameters
| Parameter | Range | Note |
|-----------|-------|------|
| Layers | 6-12 | Encoder/Decoder depth |
| Heads | 8-16 | Parallel attention |
| Model Dim | 512-1024 | Embedding size |
| FF Dim | 2048-4096 | Feed-forward size |

## 5. Mathematical Foundations

### 5.1. Attention Formula

$$\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

### 5.2. Layer Normalization

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x-\mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where:
- μ: mean
- σ: standard deviation
- γ, β: learned parameters
- ε: small constant for numerical stability

## 6. Implementation Tips

### 6.1. Best Practices
1. **Initialization**:
   - Xavier/Glorot for weights
   - Zero for biases

2. **Training**:
   - Adam optimizer
   - Learning rate warmup
   - Label smoothing

3. **Regularization**:
   - Dropout in attention
   - Layer dropout
   - Residual dropout

### 6.2. Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Vanishing Gradient | Layer normalization |
| Memory Usage | Gradient checkpointing |
| Training Time | Mixed precision training |

### 6.3. Performance Optimization
- Use efficient attention implementations
- Batch processing
- Hardware acceleration (GPU/TPU)
- Gradient accumulation for large batches

## Key Takeaways

1. **Architecture Strengths**:
   - Parallel processing
   - Global dependencies
   - Scalable attention

2. **Practical Considerations**:
   - Memory intensive
   - Requires large datasets
   - Benefits from pre-training

3. **Future Directions**:
   - Efficient attention variants
   - Sparse transformers
   - Hardware-specific optimizations

---

*Note: This document provides a high-level overview of the Transformer architecture. For implementation details, please refer to the original paper and modern implementations.*
