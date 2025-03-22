from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
from datasets import *
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from transformers import BertTokenizer, AdamW

# Define the model name and cache directory
model_name = "bert-base-uncased"
cache_dir = r"D:\VS Code Projects\bert training\hugging face\cache"
# Load the model and tokenizer
model = BertModel.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
save_path = r"D:\VS Code Projects\bert training\datasets"
raw_dataset = load_from_disk(save_path)
train_dataset = raw_dataset['train']
test_dataset = raw_dataset['test']


# define the sample ratio
sample_ratio = 0.0005

# calculate the sample size
train_size = int(len(train_dataset) * sample_ratio)
test_size = int(len(test_dataset) * sample_ratio)

# randomly sample the dataset
train_dataset, _ = random_split(train_dataset, [train_size, len(train_dataset) - train_size])
test_dataset, _ = random_split(test_dataset, [test_size, len(test_dataset) - test_size])

# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练模型
pretrained = BertModel.from_pretrained(r"D:\VS Code Projects\bert training\hugging face\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594", cache_dir=cache_dir).to(device)
print(pretrained)

# 定义下游model:进行二分类任务
# 冻结所有参数
for param in pretrained.parameters():
    param.requires_grad = False

# 只让最后两层可训练
for param in pretrained.encoder.layer[-2:].parameters():
    param.requires_grad = True

class Model(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.bert = pretrained  # 已经做了部分冻结处理
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # 这里不需要额外的 `with torch.no_grad()`，因为前几层已经被冻结
        out = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        # 取最后一层 [CLS] 向量
        out = out.last_hidden_state[:, 0]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    
epochs = 5
# 定义训练轮数
lr = 1e-5
# 定义学习率

# 对数据进行编码处理 coding the datasets
def collate_fn(data):
    sentences = [i['sentence'] for i in data]
    labels = [i['labels']['5d'] for i in data]

    # 对文本进行编码
    data = tokenizer.batch_encode_plus(
        sentences, 
        padding='max_length', 
        truncation=True, 
        max_length=512, 
        return_tensors="pt",
        return_length=True
    )

    input_ids = data["input_ids"].to(device)
    attention_mask = data["attention_mask"].to(device)
    token_type_ids = data["token_type_ids"].to(device)
    labels =torch.LongTensor(labels).to(device)
    #将原始数字label转换为torch的类型

    return input_ids, attention_mask, token_type_ids, labels

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last= True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 指定参数结果保存路径
save_dir = r"c:\Users\Kevin\OneDrive\Desktop\MQF学习资料\files of lessons\winter quarter\MGTF 423\final projects\Bert_trained\params"

if __name__ == "__main__":
    model = Model(pretrained).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
            # grad to zero
            optimizer.zero_grad()

            output = model(input_ids, attention_mask, token_type_ids)

            loss = torch.nn.functional.cross_entropy(output, labels)

            loss.backward()
            # renew the parameters
            optimizer.step()

            if i % 10 == 0:
                out = output.argmax(dim=1)
                acc = (out == labels).sum().item() / len(labels)
                print(f"epoch: {epoch}, step: {i}, loss: {loss.item()}, accuracy: {acc}")

    # Save the model state dictionary to a file
    torch.save(model.state_dict(), os.path.join(save_dir, f"{epoch}bert.pt"))
    print(epoch, "model has been saved successfully")