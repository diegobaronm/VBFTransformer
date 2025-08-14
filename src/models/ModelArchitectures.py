# === Standard imports ===
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === External libraries ===
from omegaconf import DictConfig
from loguru import logger
import lightning as L
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Custom Transformer Encoder to get out the weights from the attention each head
class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src: The input token embeddings
        # src2: The attention transformermed tokens
        
        # Call self-attention with need_weights=True
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False # To get the attention of each head
        )

        # drop out + residual connection + layer norm + linear feed forward with ReLU (inherited from base class)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

class BasicTransformer(nn.Module):
    def __init__(self, config: dict):
        super(BasicTransformer, self).__init__()

        input_dim = config['input_dim']
        nhead = config['num_heads']
        num_layers = config['num_layers']
        dnn_layers = config['input_embedder_NN_layers']
        output_activation = config['output_activation']
        pooling_type = config.get('pooling_type', 'mean').lower()  # 'mean' or 'attention'

        norm_type = config.get('norm_type', None)  # 'batchnorm', 'layernorm', or None
        dropout_prob = config.get('dropout_prob', 0.0)  # float between 0 and 1

        # === Input Embedder ===
        input_layers = []
        prev_dim = input_dim
        for dim in dnn_layers:
            input_layers.append(nn.Linear(prev_dim, dim))
            
            # Add normalization layer if specified
            if norm_type == 'batchnorm':
                input_layers.append(nn.BatchNorm1d(dim))
            elif norm_type == 'layernorm':
                input_layers.append(nn.LayerNorm(dim))

            # Add activation
            input_layers.append(nn.ReLU())

            # Add dropout if > 0
            if dropout_prob > 0:
                input_layers.append(nn.Dropout(dropout_prob))

            prev_dim = dim
        self.input_embedder = nn.Sequential(*input_layers)

        embed_dim = dnn_layers[-1]  # Last layer size

        # === Transformer === 
        # Stacks num_layers of the custom encoder layers 
        self.transformer_encoder = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                batch_first=True, 
                norm_first=True, 
                dropout=dropout_prob,
            ) for _ in range(num_layers)
        ])

        # === Output head ===
        self.output_linear = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.output_activation = output_activation.lower()
        self.pooling_type = pooling_type

        if pooling_type == 'attention':
            # Learnable attention vector for pooling [embed_dim, 1]
            self.attention_vector = nn.Parameter(torch.randn(embed_dim, 1))

    def mean_pooling(self, x):
        return x.mean(dim=1)

    def attention_pooling(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        scores = torch.matmul(x, self.attention_vector)  # batch x seq_len x 1
        weights = torch.softmax(scores, dim=1)           # softmax over seq_len
        pooled = (x * weights).sum(dim=1)                 # weighted sum over seq_len
        return pooled

    def apply_output_activation(self, x):
        if self.output_activation == 'softplus':
            return F.softplus(x)
        elif self.output_activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.output_activation == 'tanh':
            return torch.tanh(x)
        elif self.output_activation == 'relu':
            return F.relu(x)
        elif self.output_activation == 'linear':
            return x
        else:
            raise ValueError(f"Unsupported output activation: {self.output_activation}")

    def forward(self, x, return_attentions=False):
        x = self.input_embedder(x)
        all_attentions = []
        
        for layer in self.transformer_encoder:
            x, attn = layer(x)
            all_attentions.append(attn)  # shape: [batch, seq, seq]

        if self.pooling_type == 'mean':
            x = self.mean_pooling(x)
        elif self.pooling_type == 'attention':
            x = self.attention_pooling(x)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        x = self.output_linear(x)
        x = self.apply_output_activation(x)

        if return_attentions:
            return x, all_attentions
        return x

class ExtraFeatureTransformer(nn.Module):
    def __init__(self, config: dict):
        super(ExtraFeatureTransformer, self).__init__()

        input_dim = config['input_dim'] + 1
        nhead = config['num_heads']
        num_layers = config['num_layers']
        dnn_layers = config['input_embedder_NN_layers']
        output_activation = config['output_activation']
        pooling_type = config.get('pooling_type', 'mean').lower()  # 'mean' or 'attention'

        norm_type = config.get('norm_type', None)  # 'batchnorm', 'layernorm', or None
        dropout_prob = config.get('dropout_prob', 0.0)  # float between 0 and 1

        # === Input Embedder ===
        input_layers = []
        prev_dim = input_dim
        for dim in dnn_layers:
            input_layers.append(nn.Linear(prev_dim, dim))
            
            # Add normalization layer if specified
            if norm_type == 'batchnorm':
                input_layers.append(nn.BatchNorm1d(dim))
            elif norm_type == 'layernorm':
                input_layers.append(nn.LayerNorm(dim))

            # Add activation
            input_layers.append(nn.ReLU())

            # Add dropout if > 0
            if dropout_prob > 0:
                input_layers.append(nn.Dropout(dropout_prob))

            prev_dim = dim
        self.input_embedder = nn.Sequential(*input_layers)

        embed_dim = dnn_layers[-1]  # Last layer size
        
        # === Transformer === not changing dropout here
        self.transformer_encoder = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                batch_first=True, 
                norm_first=True, 
                dropout=dropout_prob,
            ) for _ in range(num_layers)
        ])

        # === Output head ===
        self.output_linear = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.output_activation = output_activation.lower()
        self.pooling_type = pooling_type

        if pooling_type == 'attention':
            # Learnable attention vector for pooling [embed_dim, 1]
            self.attention_vector = nn.Parameter(torch.randn(embed_dim, 1))

    def mean_pooling(self, x):
        return x.mean(dim=1)

    def attention_pooling(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        scores = torch.matmul(x, self.attention_vector)  # batch x seq_len x 1
        weights = torch.softmax(scores, dim=1)           # softmax over seq_len
        pooled = (x * weights).sum(dim=1)                 # weighted sum over seq_len
        return pooled

    def apply_output_activation(self, x):
        if self.output_activation == 'softplus':
            return F.softplus(x)
        elif self.output_activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.output_activation == 'tanh':
            return torch.tanh(x)
        elif self.output_activation == 'relu':
            return F.relu(x)
        elif self.output_activation == 'linear':
            return x
        else:
            raise ValueError(f"Unsupported output activation: {self.output_activation}")
    def forward(self, x, extra_feature, return_attentions=False):
        # Embed token inputs: (batch, seq_len, embed_dim)
        if extra_feature.dim() == 1:  # shape (batch,)
            extra_feature = extra_feature.unsqueeze(-1)  # shape (batch, 1)
        
        extra_expanded = extra_feature.unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, extra_expanded], dim=-1)

        x = self.input_embedder(x)
        
        # Embed event-level feature: (batch, embed_dim)
        #extra_feature = self.event_embedder(extra_feature)
       # extra_feature = extra_feature.unsqueeze(1)  # (batch, 1, embed_dim)
        
        # Concatenate event token to token sequence along seq_len dimension
       # x = torch.cat([extra_feature, x], dim=1)  # (batch, seq_len + 1, embed_dim)
        
        all_attentions = []
        
        for layer in self.transformer_encoder:
            x, attn = layer(x)
            all_attentions.append(attn)  # shape: [batch, seq_len + 1, seq_len + 1]
        
        # Pooling over entire sequence including extra token
        if self.pooling_type == 'mean':
            x = self.mean_pooling(x)
        elif self.pooling_type == 'attention':
            x = self.attention_pooling(x)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
        
        x = self.output_linear(x)
        x = self.apply_output_activation(x)
        
        if return_attentions:
            return x, all_attentions
        return x
    



class ConcatenationTransformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # configs
        inter_dim   = config['event_dim']  # e.g. 4
        part_dim    = config['input_dim']     # e.g. 8
        nhead       = config['num_heads']
        num_layers  = config['num_layers']
        dnn_layers  = config['input_embedder_NN_layers']
        norm_type   = config.get('norm_type', None)
        dropout_prob= config.get('dropout_prob', 0.0)
        pooling     = config.get('pooling_type', 'mean').lower()

        embed_dim = dnn_layers[-1] # last DNN dimension
        
        self.pooling_type = pooling
        self.norm_type = norm_type

        # --- Interaction embedder ---
        self.interaction_embedder = self._make_embedder(
            in_dim=inter_dim,
            dnn_layers=dnn_layers,
            norm_type=norm_type,
            dropout_prob=dropout_prob
        )

        # --- Particle embedder ---
        self.particle_embedder = self._make_embedder(
            in_dim=part_dim,
            dnn_layers=dnn_layers,
            norm_type=norm_type,
            dropout_prob=dropout_prob
        )

        # --- Transformer layers ---
        self.transformer_encoder = nn.ModuleList([
            CustomTransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                batch_first=True,
                norm_first=True,
                dropout=dropout_prob,
            ) for _ in range(num_layers)
        ])

        # --- Pooling attention vector if needed ---
        if pooling == 'attention':
            self.attention_vector = nn.Parameter(torch.randn(embed_dim, 1))

        # --- Final MLP head ---
        self.output_linear = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.output_activation = config.get('output_activation', 'linear').lower()

    def _make_embedder(self, in_dim, dnn_layers, norm_type, dropout_prob):
        layers = []
        prev = in_dim
        for dim in dnn_layers:
            layers.append(nn.Linear(prev, dim))
            if norm_type == 'batchnorm':
                layers.append(nn.BatchNorm1d(dim))
            elif norm_type == 'layernorm':
                layers.append(nn.LayerNorm(dim))
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            prev = dim
        return nn.Sequential(*layers)

    def mean_pooling(self, x):
        return x.mean(dim=1)

    def attention_pooling(self, x):
        scores = x @ self.attention_vector     # (B, L, 1)
        weights = torch.softmax(scores, dim=1)  # (B, L, 1)
        return (x * weights).sum(dim=1)         # (B, embed_dim)

    def apply_output_activation(self, x):
        funcs = {
            'softplus': F.softplus,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'relu': F.relu,
            'linear': lambda x: x
        }
        return funcs[self.output_activation](x)

    def forward(self, x_part, x_inter, return_attentions=False):
        # x_inter: (B, 10, inter_dim)
        # x_part : (B,  5, part_dim)

        B, L1, _ = x_inter.shape
        _, L2, _ = x_part.shape

        # embed inter tokens
        if self.norm_type == 'batchnorm':
            x1 = x_inter.reshape(-1, x_inter.size(-1))
            x1 = self.interaction_embedder(x1)
            x1 = x1.reshape(B, L1, -1)
        else:
            x1 = self.interaction_embedder(x_inter)

        # embed part tokens
        if self.norm_type == 'batchnorm':
            x2 = x_part.reshape(-1, x_part.size(-1))
            x2 = self.particle_embedder(x2)
            x2 = x2.reshape(B, L2, -1)
        else:
            x2 = self.particle_embedder(x_part)

        # concatenate along sequence dim
        x = torch.cat([x1, x2], dim=1)  # shape (B, L1+L2, embed_dim)
        assert x.size(1) > 0, "Combined sequence length is zero!"

        all_attns = []
        for layer in self.transformer_encoder:
            x, attn = layer(x)
            if return_attentions:
                all_attns.append(attn)

        # pooling
        if self.pooling_type == 'mean':
            x = self.mean_pooling(x)
        else:
            x = self.attention_pooling(x)

        # head + activation
        x = self.output_linear(x)
        x = self.apply_output_activation(x)

        if return_attentions:
            return x, all_attns
        return x
