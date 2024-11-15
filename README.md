# Fine-tuning DistilBERT cho Phân loại Văn bản Đa lớp

## 1. Giới thiệu

Bài viết này sẽ hướng dẫn cách fine-tune mô hình DistilBERT cho bài toán phân loại văn bản đa lớp.
Đây là một trong những bài toán phổ biến nhất trong lĩnh vực xử lý ngôn ngữ tự nhiên, nơi một đoạn văn bản cần được phân loại vào một trong số các danh mục cho trước.
Lý do chọn mô hình DistilBERT:
- **Nhanh nhất**
- **Yêu cầu tài nguyên thấp nhất**
- **Dễ triển khai**
- **Chi phí thấp**

### 1.1 Dataset và Công cụ

- **Dataset**: News aggregator dataset từ UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/359/news+aggregator
- **Model**: DistilBERT - một phiên bản nhỏ gọn của BERT
- **Framework**: PyTorch và Transformers

## 2. Thiết lập Môi trường

### 2.1 Cài đặt Thư viện

```python
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer

# Thiết lập GPU
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
```

### 2.2 Cấu hình Hyperparameters

```python
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
```

## 3. Xử lý Dữ liệu

### 3.1 Tải và Tiền xử lý Dữ liệu

```python
# Đọc dữ liệu
df = pd.read_csv('./data/newsCorpora.csv', sep='\t', 
                 names=['ID','TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 
                       'STORY', 'HOSTNAME', 'TIMESTAMP'])

# Lọc cột cần thiết
df = df[['TITLE','CATEGORY']]

# Mapping categories
my_dict = {
    'e': 'Entertainment',
    'b': 'Business',
    't': 'Science',
    'm': 'Health'
}

df['CATEGORY'] = df['CATEGORY'].apply(lambda x: my_dict[x])

# Encoding categories
encode_dict = {}
def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x] = len(encode_dict)
    return encode_dict[x]

df['ENCODE_CAT'] = df['CATEGORY'].apply(lambda x: encode_cat(x))
```


### 3.2 Custom Dataset Class

```python
class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        title = str(self.data.TITLE[index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.ENCODE_CAT[index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len
```

### 3.3 Tạo Train/Test Split và DataLoader

```python
# Chia dataset
train_size = 0.8
train_dataset = df.sample(frac=train_size, random_state=200)
test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

# Tạo dataset objects
training_set = Triage(train_dataset, tokenizer, MAX_LEN)
testing_set = Triage(test_dataset, tokenizer, MAX_LEN)

# Tạo dataloader
train_params = {
    'batch_size': TRAIN_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0
}

test_params = {
    'batch_size': VALID_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0
}

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)
```

## 4. Xây dựng Mô hình

### 4.1 Custom DistilBERT Model

```python
class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

# Khởi tạo model
model = DistillBERTClass()
model.to(device)
```


### 4.2 Loss Function và Optimizer

```python
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
```

## 5. Training Process

### 5.1 Training Function

```python
def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct

def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)
        
        if _%5000==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples 
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")
```


### 5.2 Validation Function

```python
def valid(model, testing_loader):
    model.eval()
    n_correct = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model(ids, mask).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)
            
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")
    
    return epoch_accu
```

## 6. Lưu Model và Tokenizer

```python
output_model_file = './models/pytorch_distilbert_news.bin'
output_vocab_file = './models/vocab_distilbert_news.bin'

torch.save(model, output_model_file)
tokenizer.save_vocabulary(output_vocab_file)
```

## 7. Best Practices và Tips

1. **Hyperparameter Tuning:**
   - Điều chỉnh batch_size dựa trên GPU memory
   - Thử nghiệm với learning rate khác nhau
   - Tăng số epochs nếu cần

2. **Data Preprocessing:**
   - Cân bằng classes trong dataset
   - Xử lý text: loại bỏ special characters, lowercase
   - Augmentation data nếu cần

3. **Model Architecture:**
   - Thử nghiệm dropout rates khác nhau
   - Điều chỉnh số neurons trong pre_classifier layer
   - Thêm batch normalization nếu cần


## Kết luận

Fine-tuning DistilBERT cho bài toán phân loại văn bản đa lớp là một quy trình có thể được thực hiện hiệu quả với PyTorch và Hugging Face Transformers. 
Mô hình đạt được độ chính xác cao trên tập validation và có thể được sử dụng cho nhiều ứng dụng thực tế.
