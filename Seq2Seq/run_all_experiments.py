import os
import sys
import json
import subprocess
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

embed_sizes = [4, 8, 16, 32, 64]

base_script_std = '''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import json

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def build_dataset():
    animals = {"duck": "鸭子", "cat": "猫", "dog": "狗", "cow": "牛", "bird": "鸟", "horse": "马", "bear": "熊", "lion": "狮子"}
    colors = {"quiet": "静静地", "red": "红色", "black": "黑色", "white": "白色", "green": "绿色", "yellow": "黄色", "blue": "蓝色", "brown": "棕色"}
    numbers = {"one": "一个", "two": "两个", "three": "三个", "four": "四个", "five": "五个", "six": "六个"}
    names = {"henry": "亨利", "fiona": "菲奥娜", "george": "乔治", "bob": "鲍勃"}
    items = {"chairs": "椅子", "tables": "桌子", "cars": "汽车", "hats": "帽子", "shoes": "鞋子"}
    
    pairs_template1 = [
        (f"{name} has {num} {color} {item}", f"{names[name]} 有 {numbers[num]} {colors[color]} {items[item]}")
        for name in names for num in numbers for color in colors for item in items
    ]
    pairs_template2 = [
        (f"the {color} {animal} is {color2}", f"那只 {colors[color]} {animals[animal]} 是 {colors[color2]}")
        for color in colors for animal in animals for color2 in colors if color != color2
    ]
    
    all_pairs = pairs_template1 + pairs_template2
    random.shuffle(all_pairs)
    return all_pairs

data_pairs = build_dataset()
train_size = int(len(data_pairs) * 0.8)
val_size = int(len(data_pairs) * 0.1)

train_data = data_pairs[:train_size]
val_data = data_pairs[train_size:train_size+val_size]
test_data = data_pairs[train_size+val_size:]

def build_vocab(data):
    en_words = set()
    zh_words = set()
    for en, zh in data:
        en_words.update(en.split())
        zh_words.update(zh.split())
    en_vocab = {w: i + 4 for i, w in enumerate(sorted(en_words))}
    en_vocab['<PAD>'] = 0
    en_vocab['<BOS>'] = 1
    en_vocab['<EOS>'] = 2
    en_vocab['<UNK>'] = 3
    
    zh_vocab = {w: i + 4 for i, w in enumerate(sorted(zh_words))}
    zh_vocab['<PAD>'] = 0
    zh_vocab['<BOS>'] = 1
    zh_vocab['<EOS>'] = 2
    zh_vocab['<UNK>'] = 3
    return en_vocab, zh_vocab

en_vocab, zh_vocab = build_vocab(data_pairs)
zh_idx2word = {i: w for w, i in zh_vocab.items()}

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

class TranslationDataset(Dataset):
    def __init__(self, data_pairs, en_vocab, zh_vocab):
        self.data_pairs = data_pairs
        self.en_vocab = en_vocab
        self.zh_vocab = zh_vocab
        
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        en, zh = self.data_pairs[idx]
        en_indices = [self.en_vocab.get(w, UNK_IDX) for w in en.split()]
        zh_indices = [self.zh_vocab.get(w, UNK_IDX) for w in zh.split()]
        
        eng_reversed = en_indices[::-1]
        pad_len = max(0, 10 - len(eng_reversed))
        eng_padded = [PAD_IDX] * pad_len + eng_reversed
        chn_target = [BOS_IDX] + zh_indices + [EOS_IDX]
        
        return torch.tensor(eng_padded, dtype=torch.long), torch.tensor(chn_target, dtype=torch.long)

def collate_fn(batch):
    eng_batch, chn_batch = zip(*batch)
    eng_padded = torch.nn.utils.rnn.pad_sequence(eng_batch, batch_first=True, padding_value=PAD_IDX)
    chn_padded = torch.nn.utils.rnn.pad_sequence(chn_batch, batch_first=True, padding_value=PAD_IDX)
    return eng_padded, chn_padded

train_dataset = TranslationDataset(train_data, en_vocab, zh_vocab)
val_dataset = TranslationDataset(val_data, en_vocab, zh_vocab)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size=128):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size=128):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        
        _, hidden = self.encoder(source)
        x = target[:, 0].unsqueeze(1)
        
        for t in range(1, target_len):
            prediction, hidden = self.decoder(x, hidden)
            outputs[:, t, :] = prediction
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            x = target[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
            
        return outputs

ENG_VOCAB_SIZE = len(en_vocab)
CHN_VOCAB_SIZE = len(zh_vocab)
HIDDEN_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(ENG_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
decoder = Decoder(CHN_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model, dataloader, optimizer, criterion, device, clip=1.0):
    model.train()
    epoch_loss = 0
    for source, target in dataloader:
        source = source.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        outputs = model(source, target, teacher_forcing_ratio=0.5)
        outputs = outputs[:, 1:].contiguous().view(-1, outputs.shape[-1])
        target = target[:, 1:].contiguous().view(-1)
        loss = criterion(outputs, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for source, target in dataloader:
            source = source.to(device)
            target = target.to(device)
            outputs = model(source, target, teacher_forcing_ratio=0.0)
            outputs = outputs[:, 1:].contiguous().view(-1, outputs.shape[-1])
            target = target[:, 1:].contiguous().view(-1)
            loss = criterion(outputs, target)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def translate(model, english_indices, device, max_len=20):
    model.eval()
    with torch.no_grad():
        eng_reversed = english_indices[::-1]
        pad_len = max(0, 10 - len(eng_reversed))
        eng_padded = [PAD_IDX] * pad_len + eng_reversed
        english_tensor = torch.tensor(eng_padded, dtype=torch.long).unsqueeze(0).to(device)
        _, hidden = model.encoder(english_tensor)
        x = torch.tensor([[BOS_IDX]], dtype=torch.long).to(device)
        predicted_words = []
        for _ in range(max_len):
            prediction, hidden = model.decoder(x, hidden)
            top1 = prediction.argmax(1).item()
            if top1 == EOS_IDX:
                break
            predicted_words.append(top1)
            x = torch.tensor([[top1]], dtype=torch.long).to(device)
    return predicted_words

EPOCHS = 100
PATIENCE = 5
best_val_loss = float("inf")
early_stop_counter = 0

train_losses = []
val_losses = []

save_path = f"best_model_{EMBED_SIZE}.pt"

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss = evaluate(model, val_loader, criterion, DEVICE)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), save_path)
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            break

model.load_state_dict(torch.load(save_path))
correct_count = 0
total_count = len(test_data)

for k in range(total_count):
    sample_en, sample_zh = test_data[k]
    sample_en_idx = [en_vocab.get(w, UNK_IDX) for w in sample_en.split()]
    pred_idx = translate(model, sample_en_idx, DEVICE)
    pred_zh = " ".join([zh_idx2word.get(i, "<UNK>") for i in pred_idx])
    if pred_zh == sample_zh:
        correct_count += 1

accuracy = (correct_count / total_count) * 100

result = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "accuracy": accuracy
}

with open(f"_res_std_{EMBED_SIZE}.json", "w", encoding="utf-8") as f:
    json.dump(result, f)
'''

base_script_attn = '''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import torch.nn.functional as F
import json

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def build_dataset():
    animals = {"duck": "鸭子", "cat": "猫", "dog": "狗", "cow": "牛", "bird": "鸟", "horse": "马", "bear": "熊", "lion": "狮子"}
    colors = {"quiet": "静静地", "red": "红色", "black": "黑色", "white": "白色", "green": "绿色", "yellow": "黄色", "blue": "蓝色", "brown": "棕色"}
    numbers = {"one": "一个", "two": "两个", "three": "三个", "four": "四个", "five": "五个", "six": "六个"}
    names = {"henry": "亨利", "fiona": "菲奥娜", "george": "乔治", "bob": "鲍勃"}
    items = {"chairs": "椅子", "tables": "桌子", "cars": "汽车", "hats": "帽子", "shoes": "鞋子"}
    
    pairs_template1 = [
        (f"{name} has {num} {color} {item}", f"{names[name]} 有 {numbers[num]} {colors[color]} {items[item]}")
        for name in names for num in numbers for color in colors for item in items
    ]
    pairs_template2 = [
        (f"the {color} {animal} is {color2}", f"那只 {colors[color]} {animals[animal]} 是 {colors[color2]}")
        for color in colors for animal in animals for color2 in colors if color != color2
    ]
    
    all_pairs = pairs_template1 + pairs_template2
    random.shuffle(all_pairs)
    return all_pairs

data_pairs = build_dataset()
train_size = int(len(data_pairs) * 0.8)
val_size = int(len(data_pairs) * 0.1)

train_data = data_pairs[:train_size]
val_data = data_pairs[train_size:train_size+val_size]
test_data = data_pairs[train_size+val_size:]

def build_vocab(data):
    en_words = set()
    zh_words = set()
    for en, zh in data:
        en_words.update(en.split())
        zh_words.update(zh.split())
    en_vocab = {w: i + 4 for i, w in enumerate(sorted(en_words))}
    en_vocab['<PAD>'] = 0
    en_vocab['<BOS>'] = 1
    en_vocab['<EOS>'] = 2
    en_vocab['<UNK>'] = 3
    
    zh_vocab = {w: i + 4 for i, w in enumerate(sorted(zh_words))}
    zh_vocab['<PAD>'] = 0
    zh_vocab['<BOS>'] = 1
    zh_vocab['<EOS>'] = 2
    zh_vocab['<UNK>'] = 3
    return en_vocab, zh_vocab

en_vocab, zh_vocab = build_vocab(data_pairs)
zh_idx2word = {i: w for w, i in zh_vocab.items()}

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

class TranslationDataset(Dataset):
    def __init__(self, data_pairs, en_vocab, zh_vocab):
        self.data_pairs = data_pairs
        self.en_vocab = en_vocab
        self.zh_vocab = zh_vocab
        
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        en, zh = self.data_pairs[idx]
        en_indices = [self.en_vocab.get(w, UNK_IDX) for w in en.split()]
        zh_indices = [self.zh_vocab.get(w, UNK_IDX) for w in zh.split()]
        
        eng_reversed = en_indices[::-1]
        pad_len = max(0, 10 - len(eng_reversed))
        eng_padded = [PAD_IDX] * pad_len + eng_reversed
        chn_target = [BOS_IDX] + zh_indices + [EOS_IDX]
        
        return torch.tensor(eng_padded, dtype=torch.long), torch.tensor(chn_target, dtype=torch.long)

def collate_fn(batch):
    eng_batch, chn_batch = zip(*batch)
    eng_padded = torch.nn.utils.rnn.pad_sequence(eng_batch, batch_first=True, padding_value=PAD_IDX)
    chn_padded = torch.nn.utils.rnn.pad_sequence(chn_batch, batch_first=True, padding_value=PAD_IDX)
    return eng_padded, chn_padded

train_dataset = TranslationDataset(train_data, en_vocab, zh_vocab)
val_dataset = TranslationDataset(val_data, en_vocab, zh_vocab)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size=128):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        return output, hidden

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, hidden, encoder_outputs):
        if hidden.dim() == 2:
            hidden_q = hidden.unsqueeze(1)
        elif hidden.shape[0] == 1 and hidden.dim() == 3:
            hidden_q = hidden.transpose(0, 1)
        else:
            hidden_q = hidden

        scores = torch.bmm(hidden_q, encoder_outputs.transpose(1, 2))
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, encoder_outputs)
        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size=128):
        super(Decoder, self).__init__()
        self.attention = Attention()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size + hidden_size, vocab_size)

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x)
        context, attn_weights = self.attention(hidden, encoder_outputs)
        gru_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(gru_input, hidden)
        output_combined = torch.cat((output, context), dim=2)
        prediction = self.fc(output_combined.squeeze(1))
        return prediction, hidden, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        
        encoder_outputs, hidden = self.encoder(source)
        x = target[:, 0].unsqueeze(1)
        
        for t in range(1, target_len):
            prediction, hidden, _ = self.decoder(x, hidden, encoder_outputs)
            outputs[:, t, :] = prediction
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            x = target[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
            
        return outputs

ENG_VOCAB_SIZE = len(en_vocab)
CHN_VOCAB_SIZE = len(zh_vocab)
HIDDEN_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(ENG_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
decoder = Decoder(CHN_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model, dataloader, optimizer, criterion, device, clip=1.0):
    model.train()
    epoch_loss = 0
    for source, target in dataloader:
        source = source.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        outputs = model(source, target, teacher_forcing_ratio=0.5)
        outputs = outputs[:, 1:].contiguous().view(-1, outputs.shape[-1])
        target = target[:, 1:].contiguous().view(-1)
        loss = criterion(outputs, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for source, target in dataloader:
            source = source.to(device)
            target = target.to(device)
            outputs = model(source, target, teacher_forcing_ratio=0.0)
            outputs = outputs[:, 1:].contiguous().view(-1, outputs.shape[-1])
            target = target[:, 1:].contiguous().view(-1)
            loss = criterion(outputs, target)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def translate(model, english_indices, device, max_len=20):
    model.eval()
    with torch.no_grad():
        eng_reversed = english_indices[::-1]
        pad_len = max(0, 10 - len(eng_reversed))
        eng_padded = [PAD_IDX] * pad_len + eng_reversed
        english_tensor = torch.tensor(eng_padded, dtype=torch.long).unsqueeze(0).to(device)
        encoder_outputs, hidden = model.encoder(english_tensor)
        x = torch.tensor([[BOS_IDX]], dtype=torch.long).to(device)
        predicted_words = []
        attn_matrix = []
        for _ in range(max_len):
            prediction, hidden, attn_weights = model.decoder(x, hidden, encoder_outputs)
            top1 = prediction.argmax(1).item()
            if top1 == EOS_IDX:
                break
            predicted_words.append(top1)
            attn_matrix.append(attn_weights.squeeze(0).squeeze(0).cpu().numpy())
            x = torch.tensor([[top1]], dtype=torch.long).to(device)
    return predicted_words, attn_matrix

EPOCHS = 100
PATIENCE = 5
best_val_loss = float("inf")
early_stop_counter = 0

train_losses = []
val_losses = []

save_path = f"best_model_attention_{EMBED_SIZE}.pt"

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss = evaluate(model, val_loader, criterion, DEVICE)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), save_path)
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            break

model.load_state_dict(torch.load(save_path))
correct_count = 0
total_count = len(test_data)

for k in range(total_count):
    sample_en, sample_zh = test_data[k]
    sample_en_idx = [en_vocab.get(w, UNK_IDX) for w in sample_en.split()]
    pred_idx, attn_matrix = translate(model, sample_en_idx, DEVICE)
    pred_zh = " ".join([zh_idx2word.get(i, "<UNK>") for i in pred_idx])
    if pred_zh == sample_zh:
        correct_count += 1

accuracy = (correct_count / total_count) * 100

result = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "accuracy": accuracy
}

with open(f"_res_attn_{EMBED_SIZE}.json", "w", encoding="utf-8") as f:
    json.dump(result, f)
'''

def run_isolated(script_base, embed_size, is_attn=False):
    code = f"EMBED_SIZE = {embed_size}\n" + script_base
    temp_file = "_temp_runner.py"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(code)
    
    prefix = "attn" if is_attn else "std"
    res_file = f"_res_{prefix}_{embed_size}.json"
    if os.path.exists(res_file):
        os.remove(res_file)
        
    subprocess.run([sys.executable, temp_file], capture_output=True, text=True)
    
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    if os.path.exists(res_file):
        with open(res_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        os.remove(res_file)
        return data
    return None

results_std = {}
results_attn = {}

print("开始通过磁盘 JSON 桥接的单进程隔离训练...")

for embed_size in embed_sizes:
    print(f"\n[运行 1/2] Standard Seq2Seq (Embed Size = {embed_size})...")
    r_std = run_isolated(base_script_std, embed_size, is_attn=False)
    results_std[embed_size] = r_std
    print(f"--> Standard (Embed={embed_size}) 最终准确率: {r_std['accuracy']:.2f}%")
    
    print(f"[运行 2/2] Attention Seq2Seq (Embed Size = {embed_size})...")
    r_attn = run_isolated(base_script_attn, embed_size, is_attn=True)
    results_attn[embed_size] = r_attn
    print(f"--> Attention (Embed={embed_size}) 最终准确率: {r_attn['accuracy']:.2f}%")

print("\n所有 10 组单独进程训练完成！正在写入 compare.ipynb...")

nb = {
    'cells': [],
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.0'}
    },
    'nbformat': 4,
    'nbformat_minor': 2
}

def add_md(text):
    nb['cells'].append({'cell_type': 'markdown', 'metadata': {}, 'source': [line + '\n' for line in text.split('\n')]})

def add_code(text):
    nb['cells'].append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [line + '\n' for line in text.split('\n')]})

add_md('# 物理进程隔离下 10 组模型对比结果\n\n通过自动化脚本为每一个 Embed Size 启动完全独立的 Python 解释器进程进行训练，彻底避免了随机数状态污染。保存权重依次为 `best_model_{size}.pt` 和 `best_model_attention_{size}.pt`。')

history_json = json.dumps({"standard": results_std, "attention": results_attn}, indent=2)

add_code(f'''import matplotlib.pyplot as plt
import json

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

history = {history_json}
embed_sizes = [4, 8, 16, 32, 64]
colors = {{4: '#1f77b4', 8: '#ff7f0e', 16: '#2ca02c', 32: '#d62728', 64: '#9467bd'}}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ----- 图1: 无 Attention 模型的 Loss 曲线 -----
ax1 = axes[0]
for embed_size in embed_sizes:
    h = history["standard"][str(embed_size)]
    epochs = range(1, len(h["train_losses"]) + 1)
    c = colors[embed_size]
    ax1.plot(epochs, h["train_losses"], linestyle='-', color=c, label=f'Embed={{embed_size}} (Train)')
    ax1.plot(epochs, h["val_losses"], linestyle='--', color=c, label=f'Embed={{embed_size}} (Val)')

ax1.set_title("标准 Seq2Seq (无 Attention) 在不同 Embed Size 下的 Loss 曲线", fontsize=12)
ax1.set_xlabel("Epoch", fontsize=10)
ax1.set_ylabel("Loss", fontsize=10)
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, linestyle=':', alpha=0.6)

# ----- 图2: 带 Attention 模型的 Loss 曲线 -----
ax2 = axes[1]
for embed_size in embed_sizes:
    h = history["attention"][str(embed_size)]
    epochs = range(1, len(h["train_losses"]) + 1)
    c = colors[embed_size]
    ax2.plot(epochs, h["train_losses"], linestyle='-', color=c, label=f'Embed={{embed_size}} (Train)')
    ax2.plot(epochs, h["val_losses"], linestyle='--', color=c, label=f'Embed={{embed_size}} (Val)')

ax2.set_title("Attention Seq2Seq 在不同 Embed Size 下的 Loss 曲线", fontsize=12)
ax2.set_xlabel("Epoch", fontsize=10)
ax2.set_ylabel("Loss", fontsize=10)
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("loss_comparison.png", dpi=300)
plt.show()

# ----- 图3: 测试集句级准确率对比图 -----
plt.figure(figsize=(10, 6))

std_accs = [history["standard"][str(e)]["accuracy"] for e in embed_sizes]
attn_accs = [history["attention"][str(e)]["accuracy"] for e in embed_sizes]

plt.plot(embed_sizes, std_accs, marker='o', linewidth=2.5, color='#d62728', label='无 Attention (Standard)')
plt.plot(embed_sizes, attn_accs, marker='s', linewidth=2.5, color='#1f77b4', label='带 Attention')

for x, y in zip(embed_sizes, std_accs):
    plt.text(x, y - 4, f"{{y:.1f}}%", ha='center', va='top', fontsize=9, color='#d62728')
    
for x, y in zip(embed_sizes, attn_accs):
    plt.text(x, y + 2, f"{{y:.1f}}%", ha='center', va='bottom', fontsize=9, color='#1f77b4')

plt.title("无 Attention vs 带 Attention 在不同 Embed Size 下的句级准确率", fontsize=13)
plt.xlabel("Embedding Size", fontsize=11)
plt.ylabel("测试集准确率 (%)", fontsize=11)
plt.xticks(embed_sizes)
plt.ylim(0, 110)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=300)
plt.show()
''')

with open("compare.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("compare.ipynb 已成功生成并写入所有实验数据！")
