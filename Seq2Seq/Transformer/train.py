# file: train.py
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import time
import numpy as np
import torch
import torch.optim as optim
from hyperparams import Hyperparams as hp
from data_load import load_train_data, load_val_data, load_en_vocab, load_zh_vocab
from AttModel import AttModel

def evaluate(model, val_X, val_Y, device, batch_size):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    num_batches = int(np.ceil(len(val_X) / float(batch_size)))
    
    with torch.no_grad():
        for i in range(num_batches):
            x = torch.LongTensor(val_X[i * batch_size : (i + 1) * batch_size]).to(device)
            y = torch.LongTensor(val_Y[i * batch_size : (i + 1) * batch_size]).to(device)
            loss, _, acc = model(x, y)
            val_loss += loss.item()
            val_acc += acc.item()
            
    return val_loss / num_batches, val_acc / num_batches

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Vocab and Data
    en2idx, idx2en = load_en_vocab()
    zh2idx, idx2zh = load_zh_vocab()
    enc_voc = len(en2idx)
    dec_voc = len(zh2idx)

    train_X, train_Y = load_train_data()
    val_X, val_Y, _, _ = load_val_data()
    print(f"Loaded {len(train_X)} train samples, {len(val_X)} val samples.")

    # Instantiate Model
    model = AttModel(hp, enc_voc, dec_voc).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)

    best_val_loss = float('inf')
    best_model_path = "best_model.pth"

    print("Start Training Mini-Transformer...")
    for epoch in range(1, hp.epochs + 1):
        model.train()
        # Shuffle train data
        indices = np.arange(len(train_X))
        np.random.shuffle(indices)
        train_X_shuffled = train_X[indices]
        train_Y_shuffled = train_Y[indices]

        num_batches = int(np.ceil(len(train_X_shuffled) / float(hp.batch_size)))
        epoch_loss = 0.0
        epoch_acc = 0.0
        t1 = time.time()

        for i in range(num_batches):
            x = torch.LongTensor(train_X_shuffled[i * hp.batch_size : (i + 1) * hp.batch_size]).to(device)
            y = torch.LongTensor(train_Y_shuffled[i * hp.batch_size : (i + 1) * hp.batch_size]).to(device)

            optimizer.zero_grad()
            loss, preds, acc = model(x, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        avg_train_loss = epoch_loss / num_batches
        avg_train_acc = epoch_acc / num_batches

        # Validation
        val_loss, val_acc = evaluate(model, val_X, val_Y, device, hp.batch_size)
        elapsed = time.time() - t1

        print(f"Epoch [{epoch:02d}/{hp.epochs}] ({elapsed:.1f}s) - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best model to {best_model_path} (Val Loss: {val_loss:.4f})")

    print("\nTraining Completed Successfully!")

if __name__ == '__main__':
    train()
