# file: hyperparams.py
# -*- coding: utf-8 -*-

class Hyperparams:
    '''Hyperparameters for Mini-Transformer'''
    # data paths
    source_train = 'preprocessed/train.en'
    target_train = 'preprocessed/train.zh'
    source_val = 'preprocessed/val.en'
    target_val = 'preprocessed/val.zh'
    source_test = 'preprocessed/test.en'
    target_test = 'preprocessed/test.zh'
    
    # training
    batch_size = 64
    lr = 0.001
    logdir = 'logdir'
    epochs = 30
    
    # model
    maxlen = 15          # Maximum number of words in a sentence
    min_cnt = 1          # words occurred less than min_cnt are encoded to <UNK>
    hidden_units = 64    # alias = C, hidden dimension
    num_blocks = 2       # number of encoder/decoder blocks
    num_heads = 2        # number of attention heads (d_k = 32)
    dropout_rate = 0.1
    sinusoid = True      # If True, use sinusoidal positional encoding
