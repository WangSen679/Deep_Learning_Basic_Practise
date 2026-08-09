# file: eval.py
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import codecs
import os
import random
import time
import numpy as np
import torch
from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_en_vocab, load_zh_vocab
from AttModel import AttModel

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

def eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "GPU" else "cpu")
    print(f"Evaluation Device: {device}")

    # Load Vocab and Test Data
    X, Sources, Targets = load_test_data()
    en2idx, idx2en = load_en_vocab()
    zh2idx, idx2zh = load_zh_vocab()
    
    enc_voc = len(en2idx)
    dec_voc = len(zh2idx)

    # Initialize Model
    model = AttModel(hp, enc_voc, dec_voc).to(device)

    # Load Pretrained Model Weights
    if not os.path.exists(args.pretrained):
        raise FileNotFoundError(f"Checkpoint file '{args.pretrained}' not found. Please run train.py first!")

    state = torch.load(args.pretrained, map_location=device)
    model.load_state_dict(state, strict=False)
    print(f"Loaded pretrained weights from {args.pretrained}.")
    
    model.eval()

    with codecs.open(args.log_path, 'w', 'utf-8') as fout:
        list_of_refs, hypotheses = [], []
        exact_matches = 0
        t1 = time.time()
        
        num_batches = int(np.ceil(len(X) / float(args.batch_size)))
        all_preds = []

        with torch.no_grad():
            for i in range(num_batches):
                x = torch.LongTensor(X[i * args.batch_size : (i + 1) * args.batch_size]).to(device)
                preds_t = torch.zeros((x.size()[0], hp.maxlen), dtype=torch.long, device=device)
                
                # Autoregressive decoding
                for j in range(hp.maxlen):
                    _, _preds, _ = model(x, preds_t)
                    preds_t[:, j] = _preds[:, j]

                preds_numpy = preds_t.cpu().numpy()
                all_preds.extend(preds_numpy)

        # Decode and Calculate Metrics
        print("\n======== 迷你 Transformer 结构化翻译测试集效果展示 (前 10 条) ========")
        for idx, (source, target, pred) in enumerate(zip(Sources, Targets, all_preds)):
            # ID 3 is </s>
            pred_words = []
            for token_id in pred:
                word = idx2zh.get(token_id, "<UNK>")
                if word == "</s>":
                    break
                if word not in ["<PAD>", "<s>"]:
                    pred_words.append(word)
            got = " ".join(pred_words).strip()

            fout.write(f"- Source: {source}\n")
            fout.write(f"- Target: {target}\n")
            fout.write(f"- Got:    {got}\n\n")
            fout.flush()

            if target.strip() == got.strip():
                exact_matches += 1

            if idx < 10:
                print(f"[{idx+1:02d}] 原始英文: {source}")
                print(f"     目标中文: {target}")
                print(f"     模型预测: {got}\n")

            ref = target.split()
            hypothesis = got.split()
            if len(ref) > 0 and len(hypothesis) > 0:
                list_of_refs.append([ref])
                hypotheses.append(hypothesis)

        temp_time = time.time() - t1
        qps = len(hypotheses) / temp_time if temp_time > 0 else 0.0
        exact_acc = (exact_matches / float(len(Sources))) * 100.0

        if HAS_NLTK:
            try:
                chencherry = SmoothingFunction()
                score = corpus_bleu(list_of_refs, hypotheses, smoothing_function=chencherry.method1)
                bleu_str = f"{100 * score:.2f}"
            except Exception:
                bleu_str = "N/A"
        else:
            bleu_str = "NLTK module not installed"

        print(f"测试用时: {temp_time:.2f}s | QPS (吞吐率): {qps:.2f}")
        print(f"测试集全句完全匹配准确率 (Exact Match Acc): {exact_acc:.2f}% ({exact_matches}/{len(Sources)})")
        print(f"BLEU Score: {bleu_str}")
        fout.write(f"Exact Match Acc = {exact_acc:.2f}%\n")
        fout.write(f"BLEU Score = {bleu_str}\n")
        print("Evaluation PASS!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mini-Transformer Evaluation.")
    parser.add_argument('--device', default='CPU', type=str, help='Device: CPU or GPU')
    parser.add_argument('--pretrained', default='best_model.pth', type=str, help='Pretrained weights path')
    parser.add_argument('--batch-size', default=64, type=int, help='Evaluation batch size')
    parser.add_argument('--log-path', default='output.txt', type=str, help='Log output file path')
    
    args = parser.parse_args()
    eval(args)
