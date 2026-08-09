# file: AttModel.py
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from modules import *
from hyperparams import Hyperparams as hp

class AttModel(nn.Module):
    def __init__(self, hp_, enc_voc, dec_voc):
        '''Attention is all you need. Mini-Transformer Architecture.'''
        super(AttModel, self).__init__()
        self.hp = hp_

        self.enc_voc = enc_voc
        self.dec_voc = dec_voc

        # Encoder
        self.enc_emb = embedding(self.enc_voc, self.hp.hidden_units, scale=True)
        if self.hp.sinusoid:
            self.enc_positional_encoding = positional_encoding(self.hp.hidden_units, zeros_pad=False, scale=False)
        else:
            self.enc_positional_encoding = embedding(self.hp.maxlen, self.hp.hidden_units, zeros_pad=False, scale=False)
        
        self.enc_dropout = nn.Dropout(self.hp.dropout_rate)

        for i in range(self.hp.num_blocks):
            self.__setattr__(
                'enc_self_attention_%d' % i, 
                multihead_attention(self.hp, self.hp.hidden_units, num_heads=self.hp.num_heads, dropout_rate=self.hp.dropout_rate, causality=False)
            )
            self.__setattr__(
                'enc_feed_forward_%d' % i, 
                feedforward(self.hp.hidden_units, num_units=[4 * self.hp.hidden_units, self.hp.hidden_units])
            )

        # Decoder
        self.dec_emb = embedding(self.dec_voc, self.hp.hidden_units, scale=True)
        if self.hp.sinusoid:
            self.dec_positional_encoding = positional_encoding(self.hp.hidden_units, zeros_pad=False, scale=False)
        else:
            self.dec_positional_encoding = embedding(self.hp.maxlen, self.hp.hidden_units, zeros_pad=False, scale=False)
            
        self.dec_dropout = nn.Dropout(self.hp.dropout_rate)
        
        for i in range(self.hp.num_blocks):
            self.__setattr__(
                'dec_self_attention_%d' % i, 
                multihead_attention(self.hp, self.hp.hidden_units, num_heads=self.hp.num_heads, dropout_rate=self.hp.dropout_rate, causality=True)
            )
            self.__setattr__(
                'dec_vanilla_attention_%d' % i, 
                multihead_attention(self.hp, self.hp.hidden_units, num_heads=self.hp.num_heads, dropout_rate=self.hp.dropout_rate, causality=False)
            )
            self.__setattr__(
                'dec_feed_forward_%d' % i, 
                feedforward(self.hp.hidden_units, num_units=[4 * self.hp.hidden_units, self.hp.hidden_units])
            )

        self.logits_layer = nn.Linear(self.hp.hidden_units, self.dec_voc)
        self.label_smoothing = label_smoothing(epsilon=0.1)

    def forward(self, x, y):
        # Shift target right and prepend <s> (ID=2)
        input_tensor = torch.ones(y[:, :1].size(), device=x.device)
        self.decoder_inputs = torch.cat([(input_tensor * 2).long(), y[:, :-1]], dim=-1) # 2: <s>

        # Encoder forward
        self.enc = self.enc_emb(x)
        if self.hp.sinusoid:
            self.enc += self.enc_positional_encoding(x)
        else:
            enc_positional = torch.unsqueeze(torch.arange(0, x.size()[1], device=x.device), 0).repeat(x.size(0), 1).long()
            self.enc += self.enc_positional_encoding(enc_positional)

        self.enc = self.enc_dropout(self.enc)
        
        for i in range(self.hp.num_blocks):
            self.enc = self.__getattr__('enc_self_attention_%d' % i)(self.enc, self.enc, self.enc)
            self.enc = self.__getattr__('enc_feed_forward_%d' % i)(self.enc)

        # Decoder forward
        self.dec = self.dec_emb(self.decoder_inputs)
        if self.hp.sinusoid:
            self.dec += self.dec_positional_encoding(self.decoder_inputs)
        else:
            dec_positional = torch.unsqueeze(torch.arange(0, self.decoder_inputs.size()[1], device=x.device), 0).repeat(self.decoder_inputs.size(0), 1).long()
            self.dec += self.dec_positional_encoding(dec_positional)

        self.dec = self.dec_dropout(self.dec)

        for i in range(self.hp.num_blocks):
            self.dec = self.__getattr__('dec_self_attention_%d' % i)(self.dec, self.dec, self.dec)
            self.dec = self.__getattr__('dec_vanilla_attention_%d' % i)(self.dec, self.enc, self.enc)
            self.dec = self.__getattr__('dec_feed_forward_%d' % i)(self.dec)

        # Final projection & Logits
        self.logits = self.logits_layer(self.dec)
        self.probs = F.softmax(self.logits, dim=-1).view(-1, self.dec_voc)
        _, self.preds = torch.max(self.logits, -1)
        
        # Accuracy
        self.istarget = (1. - y.eq(0.).float()).view(-1)
        self.acc = torch.sum(self.preds.eq(y).float().view(-1) * self.istarget) / torch.sum(self.istarget)

        # Loss with Label Smoothing
        self.y_onehot = torch.zeros(self.logits.size()[0] * self.logits.size()[1], self.dec_voc, device=x.device)
        self.y_onehot = self.y_onehot.scatter_(1, y.view(-1, 1).data, 1)
        self.y_smoothed = self.label_smoothing(self.y_onehot)

        self.loss = - torch.sum(self.y_smoothed * torch.log(self.probs + 1e-10), dim=-1)
        self.mean_loss = torch.sum(self.loss * self.istarget) / torch.sum(self.istarget)

        return self.mean_loss, self.preds, self.acc
