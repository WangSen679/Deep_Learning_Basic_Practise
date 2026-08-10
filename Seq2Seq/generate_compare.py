import json

nb = {
    'cells': [],
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.0'}
    },
    'nbformat': 4,
    'nbformat_minor': 2
}

def add_markdown(text):
    nb['cells'].append({'cell_type': 'markdown', 'metadata': {}, 'source': [line + '\n' for line in text.split('\n')]})

def add_code(text):
    nb['cells'].append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [line + '\n' for line in text.split('\n')]})

add_markdown('# Seq2Seq 有无 Attention 及不同 Embedding Size 效果对比实验\n\n本 Notebook 实现了独立的随机数隔离控制：\n- 硬性锁死统一的数据集划分与词表\n- 训练每个模型前独立重置随机种子，确保每个模型均与单独文件运行时的随机数流 100% 对齐\n- 测试 Embed Size = [4, 8, 16, 32, 64]\n\n权重依次保存为 `best_model_{size}.pt` 和 `best_model_attention_{size}.pt`。')

add_code('''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import torch.nn.functional as F

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
''')

# 1. 硬性划分并锁死数据集与 DataLoader Generator 种子
add_code('''def build_dataset():
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
''')

add_code('''PAD_IDX = 0
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
test_dataset = TranslationDataset(test_data, en_vocab, zh_vocab)

# 创建带独立 Generator 的 DataLoader 辅助函数
def get_dataloaders(seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader
''')

# 模型定义完全照搬
add_code('''# ===== 1. 无 Attention 网络 (完全照搬 Encoder_Decoder.ipynb) =====
class EncoderStandard(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        super(EncoderStandard, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        return output, hidden

class DecoderStandard(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        super(DecoderStandard, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden

class Seq2SeqStandard(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqStandard, self).__init__()
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

# ===== 2. 带 Attention 网络 (完全照搬 E_D_with_Attention.ipynb) =====
class EncoderAttn(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        super(EncoderAttn, self).__init__()
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

class DecoderAttn(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        super(DecoderAttn, self).__init__()
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

class Seq2SeqAttn(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqAttn, self).__init__()
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
''')

add_code('''ENG_VOCAB_SIZE = len(en_vocab)
CHN_VOCAB_SIZE = len(zh_vocab)
HIDDEN_SIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, dataloader, optimizer, criterion, device, clip=1.0):
    model.train()
    epoch_loss = 0
    for source, target in dataloader:
        source, target = source.to(device), target.to(device)
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
            source, target = source.to(device), target.to(device)
            outputs = model(source, target, teacher_forcing_ratio=0.0)
            outputs = outputs[:, 1:].contiguous().view(-1, outputs.shape[-1])
            target = target[:, 1:].contiguous().view(-1)
            loss = criterion(outputs, target)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def translate_standard(model, english_indices, device, max_len=20):
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

def translate_attn(model, english_indices, device, max_len=20):
    model.eval()
    with torch.no_grad():
        eng_reversed = english_indices[::-1]
        pad_len = max(0, 10 - len(eng_reversed))
        eng_padded = [PAD_IDX] * pad_len + eng_reversed
        english_tensor = torch.tensor(eng_padded, dtype=torch.long).unsqueeze(0).to(device)
        encoder_outputs, hidden = model.encoder(english_tensor)
        x = torch.tensor([[BOS_IDX]], dtype=torch.long).to(device)
        predicted_words = []
        for _ in range(max_len):
            prediction, hidden, _ = model.decoder(x, hidden, encoder_outputs)
            top1 = prediction.argmax(1).item()
            if top1 == EOS_IDX:
                break
            predicted_words.append(top1)
            x = torch.tensor([[top1]], dtype=torch.long).to(device)
    return predicted_words

def calc_accuracy(model, test_data, device, is_attention=False):
    model.eval()
    correct_count = 0
    total_count = len(test_data)
    for sample_en, sample_zh in test_data:
        sample_en_idx = [en_vocab.get(w, UNK_IDX) for w in sample_en.split()]
        if is_attention:
            pred_idx = translate_attn(model, sample_en_idx, device)
        else:
            pred_idx = translate_standard(model, sample_en_idx, device)
        pred_zh = ' '.join([zh_idx2word.get(i, '<UNK>') for i in pred_idx])
        if pred_zh == sample_zh:
            correct_count += 1
    return (correct_count / total_count) * 100
''')

# 核心训练对比逻辑：针对每一个模型训练前都精确隔离并重置状态
add_code('''embed_sizes = [4, 8, 16, 32, 64]
EPOCHS = 100
PATIENCE = 5

history = {
    'standard': {},
    'attention': {}
}

print('开始自动化对比实验 (带独立状态隔离与重置)...')

for embed_size in embed_sizes:
    # ---------------- 1. 训练 Standard Seq2Seq ----------------
    set_seed(42) # 重新重置主随机流
    train_loader, val_loader = get_dataloaders(seed=42) # 独立 Generator
    
    print(f'\\n================ 训练 Standard Seq2Seq (Embed Size={embed_size}) ================')
    enc_std = EncoderStandard(ENG_VOCAB_SIZE, embed_size, HIDDEN_SIZE)
    dec_std = DecoderStandard(CHN_VOCAB_SIZE, embed_size, HIDDEN_SIZE)
    model_std = Seq2SeqStandard(enc_std, dec_std, DEVICE).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model_std.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    save_path_std = f'best_model_{embed_size}.pt'
    train_losses_std, val_losses_std = [], []
    
    for epoch in range(EPOCHS):
        t_loss = train_epoch(model_std, train_loader, optimizer, criterion, DEVICE)
        v_loss = evaluate(model_std, val_loader, criterion, DEVICE)
        train_losses_std.append(t_loss)
        val_losses_std.append(v_loss)
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            early_stop_counter = 0
            torch.save(model_std.state_dict(), save_path_std)
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f'Standard (Embed={embed_size}) 触发 Early Stopping 于 Epoch {epoch+1}')
                break
                
    model_std.load_state_dict(torch.load(save_path_std))
    acc_std = calc_accuracy(model_std, test_data, DEVICE, is_attention=False)
    print(f'Standard (Embed={embed_size}) 最终测试集准确率: {acc_std:.2f}%')
    
    history['standard'][embed_size] = {
        'train_loss': train_losses_std,
        'val_loss': val_losses_std,
        'accuracy': acc_std
    }
    
    # ---------------- 2. 训练 Attention Seq2Seq ----------------
    set_seed(42) # 重新重置主随机流
    train_loader, val_loader = get_dataloaders(seed=42) # 独立 Generator
    
    print(f'\\n================ 训练 Attention Seq2Seq (Embed Size={embed_size}) ================')
    enc_attn = EncoderAttn(ENG_VOCAB_SIZE, embed_size, HIDDEN_SIZE)
    dec_attn = DecoderAttn(CHN_VOCAB_SIZE, embed_size, HIDDEN_SIZE)
    model_attn = Seq2SeqAttn(enc_attn, dec_attn, DEVICE).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model_attn.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    save_path_attn = f'best_model_attention_{embed_size}.pt'
    train_losses_attn, val_losses_attn = [], []
    
    for epoch in range(EPOCHS):
        t_loss = train_epoch(model_attn, train_loader, optimizer, criterion, DEVICE)
        v_loss = evaluate(model_attn, val_loader, criterion, DEVICE)
        train_losses_attn.append(t_loss)
        val_losses_attn.append(v_loss)
        
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            early_stop_counter = 0
            torch.save(model_attn.state_dict(), save_path_attn)
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f'Attention (Embed={embed_size}) 触发 Early Stopping 于 Epoch {epoch+1}')
                break
                
    model_attn.load_state_dict(torch.load(save_path_attn))
    acc_attn = calc_accuracy(model_attn, test_data, DEVICE, is_attention=True)
    print(f'Attention (Embed={embed_size}) 最终测试集准确率: {acc_attn:.2f}%')
    
    history['attention'][embed_size] = {
        'train_loss': train_losses_attn,
        'val_loss': val_losses_attn,
        'accuracy': acc_attn
    }

print('\\n所有 10 组实验训练完成！')
''')

add_code('''plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

colors = {4: '#1f77b4', 8: '#ff7f0e', 16: '#2ca02c', 32: '#d62728', 64: '#9467bd'}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ----- 图1: 无 Attention 模型的 Loss 曲线 -----
ax1 = axes[0]
for embed_size in embed_sizes:
    h = history['standard'][embed_size]
    epochs = range(1, len(h['train_loss']) + 1)
    c = colors[embed_size]
    ax1.plot(epochs, h['train_loss'], linestyle='-', color=c, label=f'Embed={embed_size} (Train)')
    ax1.plot(epochs, h['val_loss'], linestyle='--', color=c, label=f'Embed={embed_size} (Val)')

ax1.set_title('标准 Seq2Seq (无 Attention) 在不同 Embed Size 下的 Loss 曲线', fontsize=12)
ax1.set_xlabel('Epoch', fontsize=10)
ax1.set_ylabel('Loss', fontsize=10)
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, linestyle=':', alpha=0.6)

# ----- 图2: 带 Attention 模型的 Loss 曲线 -----
ax2 = axes[1]
for embed_size in embed_sizes:
    h = history['attention'][embed_size]
    epochs = range(1, len(h['train_loss']) + 1)
    c = colors[embed_size]
    ax2.plot(epochs, h['train_loss'], linestyle='-', color=c, label=f'Embed={embed_size} (Train)')
    ax2.plot(epochs, h['val_loss'], linestyle='--', color=c, label=f'Embed={embed_size} (Val)')

ax2.set_title('Attention Seq2Seq 在不同 Embed Size 下的 Loss 曲线', fontsize=12)
ax2.set_xlabel('Epoch', fontsize=10)
ax2.set_ylabel('Loss', fontsize=10)
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('loss_comparison.png', dpi=300)
plt.show()

# ----- 图3: 测试集句级准确率对比图 -----
plt.figure(figsize=(10, 6))

std_accs = [history['standard'][e]['accuracy'] for e in embed_sizes]
attn_accs = [history['attention'][e]['accuracy'] for e in embed_sizes]

plt.plot(embed_sizes, std_accs, marker='o', linewidth=2.5, color='#d62728', label='无 Attention (Standard)')
plt.plot(embed_sizes, attn_accs, marker='s', linewidth=2.5, color='#1f77b4', label='带 Attention')

for x, y in zip(embed_sizes, std_accs):
    plt.text(x, y - 4, f'{y:.1f}%', ha='center', va='top', fontsize=9, color='#d62728')
    
for x, y in zip(embed_sizes, attn_accs):
    plt.text(x, y + 2, f'{y:.1f}%', ha='center', va='bottom', fontsize=9, color='#1f77b4')

plt.title('无 Attention vs 带 Attention 在不同 Embed Size 下的句级准确率', fontsize=13)
plt.xlabel('Embedding Size', fontsize=11)
plt.ylabel('测试集准确率 (%)', fontsize=11)
plt.xticks(embed_sizes)
plt.ylim(0, 110)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300)
plt.show()
''')

with open('compare.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('compare.ipynb with Generator seed isolation generated successfully!')
