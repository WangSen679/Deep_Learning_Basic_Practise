# file: modules.py
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.parameter import Parameter

class embedding(nn.Module):
    def __init__(self, vocab_size, num_units, zeros_pad=True, scale=True):
        '''Embeds a given Variable.'''
        super(embedding, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale
        self.lookup_table = Parameter(torch.Tensor(vocab_size, num_units))
        nn.init.xavier_normal_(self.lookup_table.data)
        
        if self.zeros_pad:
            self.lookup_table.data[0, :].fill_(0)

    def forward(self, inputs):
        if self.zeros_pad:
            padding_idx = 0
        else:
            padding_idx = -1

        outputs = F.embedding(
            inputs, self.lookup_table, padding_idx, None, 2, False, False
        )
        
        if self.scale:
            outputs = outputs * (self.num_units ** 0.5)
            
        return outputs


class layer_normalization(nn.Module):
    def __init__(self, features, epsilon=1e-8):
        '''Applies layer normalization.'''
        super(layer_normalization, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta


class positional_encoding(nn.Module):
    def __init__(self, num_units, zeros_pad=True, scale=True):
        '''Sinusoidal Positional Encoding.'''
        super(positional_encoding, self).__init__()
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale

    def forward(self, inputs, y=None):
        N, T = inputs.size()[0:2]
        position_ind = torch.unsqueeze(torch.arange(0, T), 0).repeat(N, 1)
        if inputs.is_cuda:
            position_ind = position_ind.cuda().long()

        position_enc = torch.zeros(T, self.num_units)
        div_term = 10000 ** (2 * (torch.arange(0, self.num_units, 2).float()) / self.num_units)
        
        pos_seq = torch.arange(0, T).float().unsqueeze(1)
        position_enc[:, 0::2] = torch.sin(pos_seq / div_term)
        position_enc[:, 1::2] = torch.cos(pos_seq / div_term)

        lookup_table = position_enc
        if inputs.is_cuda:
            lookup_table = lookup_table.cuda()

        if self.zeros_pad:
            lookup_table = torch.cat((torch.zeros(1, self.num_units, device=inputs.device),
                                      lookup_table[1:, :]), 0)
            padding_idx = 0
        else:
            padding_idx = -1

        outputs = F.embedding(position_ind, lookup_table, padding_idx, None, 2, False, False)

        if self.scale:
            outputs = outputs * (self.num_units ** 0.5)

        return outputs


class multihead_attention(nn.Module):
    def __init__(self, hp_, num_units, num_heads=2, dropout_rate=0.1, causality=False):
        '''Applies multihead attention.'''
        super(multihead_attention, self).__init__()
        self.hp = hp_
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality

        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        
        self.output_dropout = nn.Dropout(self.dropout_rate)
        self.normalization = layer_normalization(self.num_units)

    def forward(self, queries, keys, values):
        Q = self.Q_proj(queries)  # (N, T_q, C)
        K = self.K_proj(keys)     # (N, T_k, C)
        V = self.V_proj(values)   # (N, T_v, C)

        # Split into heads and concat on batch dimension
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0) # (h*N, T_q, C/h)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0) # (h*N, T_k, C/h)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0) # (h*N, T_v, C/h)

        # Attention Score
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1)) # (h*N, T_q, T_k)

        # Scaling
        outputs = outputs / (K_.size()[-1] ** 0.5)

        # Key Masking
        key_masks = torch.sign(torch.abs(torch.sum(keys, dim=-1)))  # (N, T_k)
        key_masks = key_masks.repeat(self.num_heads, 1)  # (h*N, T_k)
        key_masks = torch.unsqueeze(key_masks, 1).repeat(1, queries.size()[1], 1)  # (h*N, T_q, T_k)

        init_tensor = torch.ones(*outputs.size(), dtype=queries.dtype, device=queries.device)
        padding = init_tensor * (-2 ** 32 + 1)

        condition = key_masks.eq(0.)
        outputs = padding * condition + outputs * (~condition)

        # Causality Masking (Look-ahead mask)
        if self.causality:
            diag_vals = torch.ones(*outputs[0, :, :].size(), dtype=queries.dtype, device=queries.device)
            tril = torch.tril(diag_vals, diagonal=0) # (T_q, T_k)
            masks = torch.unsqueeze(tril, 0).repeat(outputs.size()[0], 1, 1)  # (h*N, T_q, T_k)

            mask = torch.ones(*masks.size(), dtype=queries.dtype, device=queries.device)
            padding = mask * (-2 ** 32 + 1)

            condition = masks.eq(0.)
            outputs = padding * condition + outputs * (~condition)

        # Softmax
        outputs = F.softmax(outputs, dim=-1) # (h*N, T_q, T_k)

        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs * query_masks

        # Dropout & Weighted Sum
        outputs = self.output_dropout(outputs)
        outputs = torch.bmm(outputs, V_) # (h*N, T_q, C/h)

        # Restore Multi-head shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2) # (N, T_q, C)

        # Residual Connection & LayerNorm
        outputs += queries
        outputs = self.normalization(outputs)

        return outputs


class feedforward(nn.Module):
    def __init__(self, in_channels, num_units=[256, 64]):
        '''Point-wise feed forward net.'''
        super(feedforward, self).__init__()
        self.in_channels = in_channels
        self.num_units = num_units

        self.conv1 = nn.Sequential(
            nn.Linear(in_channels, num_units[0]),
            nn.ReLU()
        )
        self.conv2 = nn.Linear(num_units[0], num_units[1])
        self.normalization = layer_normalization(self.num_units[1])

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs += inputs
        outputs = self.normalization(outputs)
        return outputs


class label_smoothing(nn.Module):
    def __init__(self, epsilon=0.1):
        '''Applies label smoothing.'''
        super(label_smoothing, self).__init__()
        self.epsilon = epsilon

    def forward(self, inputs):
        K = inputs.size()[-1]
        return ((1 - self.epsilon) * inputs) + (self.epsilon / K)
