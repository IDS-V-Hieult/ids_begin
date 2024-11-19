# Fine-tuning Llama 3.2 với Unsloth: Hướng dẫn Toàn diện

## 1. Giới thiệu

Trong bài viết này, chúng ta sẽ tìm hiểu cách fine-tune mô hình Llama 3.2 sử dụng thư viện Unsloth trên Google Colab. Unsloth là một thư viện giúp tăng tốc quá trình fine-tuning lên gấp 2 lần và tối ưu hóa việc sử dụng bộ nhớ.

## 2. Thiết lập Môi trường

### 2.1 Cài đặt Unsloth

```python
!pip install unsloth
# Cài đặt phiên bản mới nhất từ GitHub
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
```

### 2.2 Cấu hình Model

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048  # Có thể chọn bất kỳ giá trị nào
dtype = None  # None để tự động phát hiện. Float16 cho Tesla T4, V100, Bfloat16 cho Ampere+
load_in_4bit = True  # Sử dụng lượng tử hóa 4-bit để giảm bộ nhớ

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```

## 3. Thêm LoRA Adapters

LoRA (Low-Rank Adaptation) cho phép chúng ta chỉ cập nhật 1-10% tham số của mô hình:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # Có thể chọn: 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)
```

## 4. Chuẩn bị Dữ liệu

### 4.1 Cấu hình Chat Template

```python
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(
        convo, 
        tokenize = False, 
        add_generation_prompt = False
    ) for convo in convos]
    return {"text": texts}

# Load dataset
from datasets import load_dataset
dataset = load_dataset("mlabonne/FineTome-100k", split="train")
```

### 4.2 Chuẩn hóa Format ShareGPT

```python
from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched=True)
```

## 5. Training Configuration

### 5.1 Thiết lập SFTTrainer

```python
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)
```

### 5.2 Train trên Assistant Responses

```python
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
```

## 6. Inference

```python
# Enable fast inference
FastLanguageModel.for_inference(model)

messages = [
    {"role": "user", "content": "Your question here"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

# Regular inference
outputs = model.generate(
    input_ids = inputs, 
    max_new_tokens = 64,
    use_cache = True,
    temperature = 1.5,
    min_p = 0.1
)

# Streaming inference
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
outputs = model.generate(
    input_ids = inputs,
    streamer = text_streamer,
    max_new_tokens = 128,
    use_cache = True,
    temperature = 1.5,
    min_p = 0.1
)
```

## 7. Lưu và Tải Model

### 7.1 Lưu LoRA Adapters

```python
# Lưu locally
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Lưu lên Hugging Face Hub
model.push_to_hub("your_name/lora_model", token="your-token")
tokenizer.push_to_hub("your_name/lora_model", token="your-token")
```

### 7.2 Chuyển đổi sang GGUF Format

```python
# Lưu sang 8-bit Q8_0
model.save_pretrained_gguf("model", tokenizer)

# Lưu sang 16-bit
model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")

# Lưu sang Q4_K_M
model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")

# Lưu nhiều format cùng lúc
model.push_to_hub_gguf(
    "your-username/model",
    tokenizer,
    quantization_method=["q4_k_m", "q8_0", "q5_k_m"],
    token="your-token"
)
```

## 8. Best Practices

1. **Memory Management:**
   - Sử dụng 4-bit quantization để giảm memory usage
   - Điều chỉnh batch size phù hợp với GPU memory
   - Sử dụng gradient checkpointing khi cần

2. **Training Configuration:**
   - Điều chỉnh learning rate phù hợp (thường 1e-4 đến 3e-4)
   - Sử dụng warmup steps để ổn định training
   - Theo dõi loss để tránh overfitting

3. **Model Saving:**
   - Lưu checkpoints thường xuyên
   - Chọn format phù hợp với use case
   - Test model sau khi convert sang các format khác nhau

4. **Inference Optimization:**
   - Sử dụng streaming cho response dài
   - Điều chỉnh temperature và min_p cho kết quả tốt nhất
   - Đặt max_new_tokens phù hợp

## Kết luận

Fine-tuning Llama 3.2 với Unsloth mang lại nhiều lợi ích về hiệu suất và tài nguyên. Với các công cụ và kỹ thuật được trình bày trong bài viết này, bạn có thể dễ dàng tùy chỉnh mô hình cho các use case cụ thể của mình.
