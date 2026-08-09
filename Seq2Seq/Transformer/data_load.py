# file: data_load.py
# -*- coding: utf-8 -*-
from __future__ import print_function
from hyperparams import Hyperparams as hp
import numpy as np
import codecs
import os

def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1]) >= hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_zh_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/zh.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1]) >= hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(source_sents, target_sents): 
    en2idx, idx2en = load_en_vocab()
    zh2idx, idx2zh = load_zh_vocab()
    
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        # 1: <UNK>, 3: </s>
        x = [en2idx.get(word, 1) for word in (source_sent + " </s>").split()] 
        y = [zh2idx.get(word, 1) for word in (target_sent + " </s>").split()] 
        if len(x) <= hp.maxlen and len(y) <= hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)
    
    # Pad      
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.pad(x, (0, hp.maxlen - len(x)), 'constant', constant_values=(0, 0))
        Y[i] = np.pad(y, (0, hp.maxlen - len(y)), 'constant', constant_values=(0, 0))
    
    return X, Y, Sources, Targets

def load_train_data():
    en_sents = [line.strip() for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n") if line.strip()]
    zh_sents = [line.strip() for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n") if line.strip()]
    X, Y, _, _ = create_data(en_sents, zh_sents)
    return X, Y

def load_val_data():
    en_sents = [line.strip() for line in codecs.open(hp.source_val, 'r', 'utf-8').read().split("\n") if line.strip()]
    zh_sents = [line.strip() for line in codecs.open(hp.target_val, 'r', 'utf-8').read().split("\n") if line.strip()]
    X, Y, Sources, Targets = create_data(en_sents, zh_sents)
    return X, Y, Sources, Targets

def load_test_data():
    en_sents = [line.strip() for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n") if line.strip()]
    zh_sents = [line.strip() for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n") if line.strip()]
    X, Y, Sources, Targets = create_data(en_sents, zh_sents)
    return X, Sources, Targets
