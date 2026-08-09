# file: visualize.py
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import torch
import torch.nn.functional as F
import numpy as np
from hyperparams import Hyperparams as hp
from data_load import load_en_vocab, load_zh_vocab
from AttModel import AttModel

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TransformerTracer:
    '''Tracer that attaches hooks to inspect intermediate tensors and attention maps.'''
    def __init__(self, model, en_idx2word, zh_idx2word):
        self.model = model
        self.en_idx2word = en_idx2word
        self.zh_idx2word = zh_idx2word
        self.attn_maps = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        # Hook Encoder Blocks
        for i in range(self.model.hp.num_blocks):
            enc_attn = getattr(self.model, f'enc_self_attention_{i}')
            self.hooks.append(enc_attn.register_forward_hook(self._make_attn_hook(f'Encoder_Block_{i}_Self_Attn')))

        # Hook Decoder Blocks
        for i in range(self.model.hp.num_blocks):
            dec_self_attn = getattr(self.model, f'dec_self_attention_{i}')
            self.hooks.append(dec_self_attn.register_forward_hook(self._make_attn_hook(f'Decoder_Block_{i}_Self_Attn')))

            dec_cross_attn = getattr(self.model, f'dec_vanilla_attention_{i}')
            self.hooks.append(dec_cross_attn.register_forward_hook(self._make_attn_hook(f'Decoder_Block_{i}_Cross_Attn')))

    def _make_attn_hook(self, name):
        def hook(module, inputs, output):
            queries, keys, values = inputs
            
            Q = module.Q_proj(queries)
            K = module.K_proj(keys)
            V = module.V_proj(values)
            
            num_heads = module.num_heads
            Q_ = torch.cat(torch.chunk(Q, num_heads, dim=2), dim=0) # (h*N, T_q, C/h)
            K_ = torch.cat(torch.chunk(K, num_heads, dim=2), dim=0) # (h*N, T_k, C/h)
            V_ = torch.cat(torch.chunk(V, num_heads, dim=2), dim=0) # (h*N, T_v, C/h)

            raw_scores = torch.bmm(Q_, K_.permute(0, 2, 1)) / (K_.size()[-1] ** 0.5)

            # Apply Key Masking
            key_masks = torch.sign(torch.abs(torch.sum(keys, dim=-1)))  # (N, T_k)
            key_masks = key_masks.repeat(num_heads, 1)  # (h*N, T_k)
            key_masks = torch.unsqueeze(key_masks, 1).repeat(1, queries.size()[1], 1)  # (h*N, T_q, T_k)

            init_tensor = torch.ones(*raw_scores.size(), dtype=queries.dtype, device=queries.device)
            padding = init_tensor * (-2 ** 32 + 1)
            condition = key_masks.eq(0.)
            masked_scores = padding * condition + raw_scores * (~condition)

            # Apply Causality Masking if enabled
            if module.causality:
                diag_vals = torch.ones(*raw_scores[0, :, :].size(), dtype=queries.dtype, device=queries.device)
                tril = torch.tril(diag_vals, diagonal=0)
                masks = torch.unsqueeze(tril, 0).repeat(raw_scores.size()[0], 1, 1)
                condition = masks.eq(0.)
                masked_scores = padding * condition + masked_scores * (~condition)

            # Softmax
            probs = torch.softmax(masked_scores, dim=-1)

            self.attn_maps[name] = {
                'queries': queries.detach().cpu().numpy(),
                'keys': keys.detach().cpu().numpy(),
                'values': values.detach().cpu().numpy(),
                'Q': Q.detach().cpu().numpy(),
                'K': K.detach().cpu().numpy(),
                'V': V.detach().cpu().numpy(),
                'raw_scores': raw_scores.detach().cpu().numpy(),
                'probs': probs.detach().cpu().numpy()
            }
        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def print_tensor_slice(tensor_2d, row_names, title, max_cols=6):
    '''Prints actual numerical float values of a 2D tensor (showing first max_cols dimensions).'''
    print(f"\n=========================================================================")
    print(f"  {title} (展示前 {max_cols} 维数值)")
    print(f"=========================================================================")
    col_headers = "".join([f"{f'dim_{c}':>9}" for c in range(max_cols)])
    header = f"{'Token':<10}" + col_headers
    print(header)
    print("-" * len(header))

    for r_idx, name in enumerate(row_names):
        row_str = f"{name[:9]:<10}"
        for c_idx in range(max_cols):
            val = tensor_2d[r_idx, c_idx]
            row_str += f"{val:9.3f}"
        print(row_str)
    print("-" * len(header))


def print_formatted_matrix(matrix, row_labels, col_labels, title):
    '''Prints a beautifully aligned ASCII representation of an attention matrix in the terminal.'''
    print(f"\n=========================================================================")
    print(f"  {title}")
    print(f"=========================================================================")
    
    col_headers = "".join([f"{col[:7]:>9}" for col in col_labels])
    header = f"{'Token':<10}" + col_headers
    print(header)
    print("-" * len(header))

    for r_idx, row_label in enumerate(row_labels):
        row_str = f"{row_label[:9]:<10}"
        for c_idx in range(len(col_labels)):
            val = matrix[r_idx, c_idx]
            row_str += f"{val:9.3f}"
        print(row_str)
    print("-" * len(header))


def plot_attention_heatmap(attn_matrix, row_labels, col_labels, title, filename):
    '''Plots and saves a high-res heatmap PNG image for attention matrix.'''
    if not HAS_MATPLOTLIB:
        return
    try:
        fig, ax = plt.subplots(figsize=(8, 7))
        cax = ax.matshow(attn_matrix, cmap='viridis')
        fig.colorbar(cax)

        ax.set_xticks(range(len(col_labels)))
        ax.set_yticks(range(len(row_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha='left', fontsize=11)
        ax.set_yticklabels(row_labels, fontsize=11)

        ax.set_title(title, fontsize=13, pad=25)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"  [图片保存] 注意力热力图已成功生成并保存至: {filename}")
    except Exception as e:
        print(f"  (绘制图片失败: {e})")


def inspect_token_probability_generation(model, x, idx2zh, sentence_en):
    '''Traces the step-by-step conversion from Decoder Attention output to Vocabulary Logits & Probabilities.'''
    device = x.device
    print("\n=========================================================================")
    print("【第五阶段：注意力隐藏向量 -> 全连接分类层(Linear) -> 51个词表概率的分布演变】")
    print("=========================================================================")

    preds_t = torch.zeros((1, hp.maxlen), dtype=torch.long, device=device)
    current_tokens = ['<s>']

    for j in range(hp.maxlen):
        with torch.no_grad():
            # Run model forward up to logits
            _, _preds, _ = model(x, preds_t)
            
            # Extract final Decoder hidden features before logits
            # model.dec shape is (1, 15, 64)
            dec_hidden_j = model.dec[0, j] # (64,)
            
            # Extract Logits for step j
            logits_j = model.logits[0, j] # (51,)
            probs_j = F.softmax(logits_j, dim=-1) # (51,)

            selected_id = _preds[0, j].item()
            selected_word = idx2zh.get(selected_id, "<UNK>")

            preds_t[0, j] = selected_id

            if selected_word == "</s>":
                break

            # Top 5 candidate words in vocabulary
            top5_probs, top5_indices = torch.topk(probs_j, k=5)
            top5_logits = logits_j[top5_indices]

            print(f"\n> [自回归 Step {j+1:02d}] 上下文中文输入: {' '.join(current_tokens)}")
            print(f"   ├─ 注意力抽取后的 64 维隐藏向量 h (前 6 维): {dec_hidden_j[:6].cpu().numpy().round(3).tolist()}")
            print(f"   └─ 经过 nn.Linear(64, 51) 投影后，词表中 Top 5 最具竞争力的候选词概率分布:")
            
            print(f"      {'候选词':<10}{'Logit 得分':<14}{'Softmax 最终预测概率':<20}{'决策'}")
            print(f"      {'-'*55}")
            for rank in range(5):
                cand_word = idx2zh.get(top5_indices[rank].item(), "<UNK>")
                logit_val = top5_logits[rank].item()
                prob_val = top5_probs[rank].item() * 100.0
                tag = "<-- 选中 (Argmax)" if rank == 0 else ""
                print(f"      {cand_word:<10}{logit_val:<14.3f}{prob_val:<19.2f}% {tag}")
            
            current_tokens.append(selected_word)

    print("\n-------------------------------------------------------------------------")
    print(f" 最终生成的中文句: {' '.join(current_tokens)}")


def inspect_single_sentence(sentence_en):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for visualization: {device}")

    en2idx, idx2en = load_en_vocab()
    zh2idx, idx2zh = load_zh_vocab()

    enc_voc = len(en2idx)
    dec_voc = len(zh2idx)

    model = AttModel(hp, enc_voc, dec_voc).to(device)
    pretrained_path = "best_model.pth"
    if not os.path.exists(pretrained_path):
        print(f"Error: {pretrained_path} not found. Please train the model first.")
        return

    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model.eval()

    en_tokens = (sentence_en.strip() + " </s>").split()
    en_indices = [en2idx.get(w, 1) for w in en_tokens]
    
    en_indices = en_indices[:hp.maxlen]
    pad_len = hp.maxlen - len(en_indices)
    en_padded = en_indices + [0] * pad_len
    x = torch.LongTensor([en_padded]).to(device)

    tracer = TransformerTracer(model, idx2en, idx2zh)

    print(f"\n[输入测试句子] 英文原句: '{sentence_en}'")
    print(f"Token 序列 (长 {len(en_tokens)}): {en_tokens}")

    # 1. Feature Flow
    with torch.no_grad():
        enc_emb = model.enc_emb(x)
        enc_pos = model.enc_positional_encoding(x)
        enc_sum = enc_emb + enc_pos

        print("\n-------------------------------------------------------------------------")
        print("【第一阶段：特征向量具体数值展现 (前 6 维浮点数)】")
        print("-------------------------------------------------------------------------")
        
        enc_emb_np = enc_emb[0].cpu().numpy()[:len(en_tokens)]
        enc_pos_np = enc_pos[0].cpu().numpy()[:len(en_tokens)]
        enc_sum_np = enc_sum[0].cpu().numpy()[:len(en_tokens)]

        print_tensor_slice(enc_emb_np, en_tokens, "1. 词嵌入向量 (Token Embedding - 乘以 sqrt(64))")
        print_tensor_slice(enc_pos_np, [f"p_{i}_{t}" for i, t in enumerate(en_tokens)], "2. 正弦位置编码向量 (Positional Encoding)")
        print_tensor_slice(enc_sum_np, en_tokens, "3. 融合后向量 (Embedding + Positional Encoding)")

        # Autoregressive decoding
        preds_t = torch.zeros((1, hp.maxlen), dtype=torch.long, device=device)
        for j in range(hp.maxlen):
            _, _preds, _ = model(x, preds_t)
            preds_t[:, j] = _preds[:, j]

        pred_ids = preds_t[0].cpu().numpy()
        pred_words = []
        for pid in pred_ids:
            w = idx2zh.get(pid, "<UNK>")
            if w == "</s>":
                pred_words.append("</s>")
                break
            if w not in ["<PAD>", "<s>"]:
                pred_words.append(w)

        zh_tokens = ['<s>'] + pred_words

        print("\n-------------------------------------------------------------------------")
        print("【第二阶段：Decoder 生成结果】")
        print("-------------------------------------------------------------------------")
        print(f" 最终生成的中文 Token 序列: {zh_tokens}")

    # 2. Q, K, V Numerical Slices
    print("\n-------------------------------------------------------------------------")
    print("【第三阶段：Encoder Q, K, V 矩阵具体数值抽样切片】")
    print("-------------------------------------------------------------------------")
    
    enc_key = 'Encoder_Block_0_Self_Attn'
    if enc_key in tracer.attn_maps:
        Q_np = tracer.attn_maps[enc_key]['Q'][0][:len(en_tokens)]
        K_np = tracer.attn_maps[enc_key]['K'][0][:len(en_tokens)]
        V_np = tracer.attn_maps[enc_key]['V'][0][:len(en_tokens)]

        print_tensor_slice(Q_np, en_tokens, "Encoder Block 0 - 投影后的 Q 向量")
        print_tensor_slice(K_np, en_tokens, "Encoder Block 0 - 投影后的 K 向量")
        print_tensor_slice(V_np, en_tokens, "Encoder Block 0 - 投影后的 V 向量")

    # 3. Print & Plot ALL THREE Attention Types!
    print("\n-------------------------------------------------------------------------")
    print("【第四阶段：三大 Attention 模块热力图与完整对齐概率矩阵】")
    print("-------------------------------------------------------------------------")

    if 'Encoder_Block_0_Self_Attn' in tracer.attn_maps:
        probs = tracer.attn_maps['Encoder_Block_0_Self_Attn']['probs'][0][:len(en_tokens), :len(en_tokens)]
        raw = tracer.attn_maps['Encoder_Block_0_Self_Attn']['raw_scores'][0][:len(en_tokens), :len(en_tokens)]

        print_formatted_matrix(
            raw, en_tokens, en_tokens,
            title="【1. Encoder 内部自注意力原始点积得分 Q*K^T / sqrt(32)】(英文 vs 英文)"
        )
        print_formatted_matrix(
            probs, en_tokens, en_tokens,
            title="【1. Encoder 内部自注意力 Softmax 概率矩阵 (Head 0)】(英文 vs 英文)"
        )
        plot_attention_heatmap(
            probs, en_tokens, en_tokens,
            title="Encoder Self-Attention (Head 0)",
            filename="1_encoder_self_attention.png"
        )

    if 'Decoder_Block_0_Self_Attn' in tracer.attn_maps:
        probs = tracer.attn_maps['Decoder_Block_0_Self_Attn']['probs'][0][:len(zh_tokens), :len(zh_tokens)]
        plot_attention_heatmap(
            probs, zh_tokens, zh_tokens,
            title="Decoder Masked Self-Attention (Head 0)",
            filename="2_decoder_self_attention.png"
        )

    if 'Decoder_Block_0_Cross_Attn' in tracer.attn_maps:
        probs = tracer.attn_maps['Decoder_Block_0_Cross_Attn']['probs'][0][:len(zh_tokens), :len(en_tokens)]
        raw = tracer.attn_maps['Decoder_Block_0_Cross_Attn']['raw_scores'][0][:len(zh_tokens), :len(en_tokens)]

        print_formatted_matrix(
            probs, zh_tokens, en_tokens,
            title="【3. Decoder 交叉注意力 Softmax 对齐概率矩阵 (Head 0)】(中文 Q vs 英文 K)"
        )
        plot_attention_heatmap(
            probs, zh_tokens, en_tokens,
            title="Decoder Cross-Attention Alignment (Head 0)",
            filename="3_decoder_cross_attention.png"
        )

    # 4. Step-by-step Token Probability Generation
    inspect_token_probability_generation(model, x, idx2zh, sentence_en)

    tracer.remove_hooks()
    print("\n全套可视化与数值追踪分析完成！热力图图片已保存至项目根目录。")


if __name__ == '__main__':
    sample_sentence = "there is a red horse behind the box"
    inspect_single_sentence(sample_sentence)
