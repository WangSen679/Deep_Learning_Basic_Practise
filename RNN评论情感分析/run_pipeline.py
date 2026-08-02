import os
import re
import collections
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# 设置随机种子保证可复现性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 1. 文本预处理与分词
def clean_text(text):
    if not isinstance(text, str):
        return []
    text = re.sub(r'<br\s*/?>', ' ', text)  # 去除 HTML 标签 <br />
    text = re.sub(r'[^a-zA-Z]', ' ', text)   # 仅保留英文字母
    tokens = text.lower().split()          # 转小写并按空格切分
    return tokens

def build_vocab(tokenized_texts, max_vocab_size=15000):
    counter = collections.Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    
    # 0: <PAD>, 1: <UNK>
    vocab = {'<PAD>': 0, '<UNK>': 1}
    most_common = counter.most_common(max_vocab_size - 2)
    for word, _ in most_common:
        vocab[word] = len(vocab)
    return vocab

def encode_and_pad(tokens, vocab, max_len=200):
    seq = [vocab.get(word, vocab['<UNK>']) for word in tokens]
    if len(seq) < max_len:
        seq = seq + [vocab['<PAD>']] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

# 2. 预训练词向量矩阵加载
def load_pretrained_embeddings(vocab, embed_dim=128, glove_path=None):
    vocab_size = len(vocab)
    embedding_matrix = np.random.normal(scale=1.0 / np.sqrt(embed_dim), size=(vocab_size, embed_dim))
    embedding_matrix[vocab['<PAD>']] = np.zeros(embed_dim)
    
    if glove_path and os.path.exists(glove_path):
        print(f"正在从 {glove_path} 加载 GloVe 词向量...")
        loaded_count = 0
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                if word in vocab:
                    vector = np.array(parts[1:], dtype=np.float32)
                    if len(vector) == embed_dim:
                        embedding_matrix[vocab[word]] = vector
                        loaded_count += 1
        print(f"成功加载 {loaded_count}/{vocab_size} 个预训练词向量。")
    else:
        print("未找到本地 GloVe 词向量文件，使用符合标准方差缩放的高斯分布初始化。")
        
    return torch.tensor(embedding_matrix, dtype=torch.float32)

# 3. PyTorch Dataset 定义
class MovieReviewDataset(Dataset):
    def __init__(self, tokenized_texts, labels, vocab, max_len=200):
        self.samples = [encode_and_pad(tokens, vocab, max_len) for tokens in tokenized_texts]
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)

# 4. 双向 LSTM 网络结构 (配合全局最大池化 Global Max Pooling)
class SentimentBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, dropout=0.5, pad_idx=0, pretrained_weight=None):
        super(SentimentBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(pretrained_weight)
            
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x 形状: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)
        
        # 🌟 核心改进：全局最大池化 (Global Max Pooling)
        # 提取每个样本在序列全长 200 个时间步上的显著特征，防止尾部 <PAD> 填充字符干扰
        out_pooled = torch.max(lstm_out, dim=1)[0]  # (batch_size, hidden_dim * 2)
        
        out = self.dropout(out_pooled)
        logits = self.fc(out).squeeze(1)
        return logits

# 别名兼容
SentimentLSTM = SentimentBiLSTM
SentimentRNN = SentimentBiLSTM

# 5. 早停机制 (Early Stopping)
class EarlyStopping:
    def __init__(self, patience=4, verbose=True, save_path='data/best_model.pt'):
        self.patience = patience
        self.verbose = verbose
        self.save_path = save_path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping 计数: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(model.state_dict(), self.save_path)
        if self.verbose:
            print(f"验证集 Loss 显著改善，已自动保存最佳权重至 {self.save_path}")

# 6. 数据均衡加载 (支持全量数据加载 sample_per_class=None)
def load_balanced_data(file_path, sample_per_class=None, random_seed=42):
    df = pd.read_csv(file_path)
    df_pos = df[df['真实标签'] == 1]
    df_neg = df[df['真实标签'] == 0]
    
    if sample_per_class is not None:
        pos_sampled = df_pos.sample(n=min(sample_per_class, len(df_pos)), random_state=random_seed)
        neg_sampled = df_neg.sample(n=min(sample_per_class, len(df_neg)), random_state=random_seed)
    else:
        # 使用全量数据，若正负类数量不一致按较小类别 1:1 对齐
        min_class_len = min(len(df_pos), len(df_neg))
        pos_sampled = df_pos.sample(n=min_class_len, random_state=random_seed)
        neg_sampled = df_neg.sample(n=min_class_len, random_state=random_seed)
        
    df_balanced = pd.concat([pos_sampled, neg_sampled]).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return df_balanced

# 7. 单条文本推理 Demo
def predict_sentiment(review_text, model, vocab, device, max_len=200):
    model.eval()
    tokens = clean_text(review_text)
    encoded = encode_and_pad(tokens, vocab, max_len)
    tensor_input = torch.tensor([encoded], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logit = model(tensor_input)
        prob = torch.sigmoid(logit).item()
        sentiment = "正面评价 (Positive)" if prob >= 0.5 else "负面评价 (Negative)"
        
    print(f"评论内容: {review_text}")
    print(f"情感预测: {sentiment} (正面置信度: {prob*100:.2f}%)\n")

# 8. 主运行程序
def main():
    train_path = "data/data_train.csv"
    test_path = "data/data_test.csv"
    output_path = "data/result_prediction.csv"
    model_save_path = "data/best_model.pt"

    print("--- 1. 扩增加载全量均衡数据集 ---")
    # sample_per_class=None 表示加载 CSV 中的全量数据
    df_raw_train = load_balanced_data(train_path, sample_per_class=None)
    df_test = load_balanced_data(test_path, sample_per_class=None)

    # 划分 85% 训练集与 15% 验证集
    df_train, df_val = train_test_split(df_raw_train, test_size=0.15, random_state=42, stratify=df_raw_train['真实标签'])
    print(f"训练集总样本数: {len(df_train)} | 验证集样本数: {len(df_val)} | 测试集样本数: {len(df_test)}")

    print("\n--- 2. 文本清洗与词表构建 ---")
    train_tokens = [clean_text(text) for text in df_train['影评内容']]
    val_tokens = [clean_text(text) for text in df_val['影评内容']]
    test_tokens = [clean_text(text) for text in df_test['影评内容']]

    vocab = build_vocab(train_tokens, max_vocab_size=15000)
    print(f"词表构建完成，大小为: {len(vocab)}")

    pretrained_weight = load_pretrained_embeddings(vocab, embed_dim=128, glove_path="data/glove.6B.100d.txt")

    max_len = 200
    batch_size = 64

    train_dataset = MovieReviewDataset(train_tokens, df_train['真实标签'].values, vocab, max_len)
    val_dataset = MovieReviewDataset(val_tokens, df_val['真实标签'].values, vocab, max_len)
    test_dataset = MovieReviewDataset(test_tokens, df_test['真实标签'].values, vocab, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用计算设备: {device}")

    # 3. 初始化双向 LSTM + 全局最大池化模型
    model = SentimentBiLSTM(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.5,
        pad_idx=0,
        pretrained_weight=pretrained_weight
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    early_stopping = EarlyStopping(patience=4, verbose=True, save_path=model_save_path)

    epochs = 20
    print("\n--- 3. 开始模型训练与验证监控 ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = criterion(logits, targets)

                val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")

        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print("触发 Early Stopping，训练提前结束！")
            break

    print("\n--- 4. 载入最佳权重并在测试集终极评估 ---")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    test_preds, test_probs = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            test_probs.extend(probs)
            test_preds.extend(preds)

    test_labels = df_test['真实标签'].values
    test_acc = accuracy_score(test_labels, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary')
    cm = confusion_matrix(test_labels, test_preds)

    print("=== 测试集评估结果 (最佳 Checkpoint 模式) ===")
    print(f"准确率 (Accuracy) : {test_acc*100:.2f}%")
    print(f"精确率 (Precision): {prec*100:.2f}%" if 'prec' in locals() else f"精确率 (Precision): {precision*100:.2f}%")
    print(f"召回率 (Recall)   : {recall*100:.2f}%")
    print(f"F1 得分 (F1 Score): {f1*100:.2f}%")
    print("混淆矩阵 (Confusion Matrix):\n", cm)

    df_result = df_test.copy()
    df_result['预测标签'] = test_preds
    df_result['预测概率'] = np.round(test_probs, 4)
    df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n预测结果已成功写入 {output_path}")

    print("\n--- 5. 单条评论情感推理测试 Demo ---")
    review_pos = "This movie was fantastic! The acting was brilliant and the story kept me on the edge of my seat."
    review_neg = "Terrible movie, waste of time. The plot was non-existent and performance was super boring."
    predict_sentiment(review_pos, model, vocab, device)
    predict_sentiment(review_neg, model, vocab, device)

if __name__ == '__main__':
    main()
