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

# === Local utility imports ===
from src.utils.Plotting import plot_metrics, permutation_feature_importance, compute_feature_importance_and_correlation_plot
from src.utils.LossFunctions import WeightedTailLoss, QuantileAwareLoss, InverseGaussianWeightedLoss, build_loss_function


class BasicTransformer(nn.Module):
    def __init__(self, config: dict):
        super(BasicTransformer, self).__init__()

        input_dim = config['input_dim']
        nhead = config['num_heads']
        num_layers = config['num_layers']
        dnn_layers = config['input_embedder_NN_layers']
        output_activation = config['output_activation']
        n_tokens = config['n_tokens']
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
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True),
            num_layers=num_layers,
            enable_nested_tensor=False
        )

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

    def forward(self, x):
        x = self.input_embedder(x)
        x = self.transformer_encoder(x)

        if self.pooling_type == 'mean':
            x = self.mean_pooling(x)
        elif self.pooling_type == 'attention':
            x = self.attention_pooling(x)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        x = self.output_linear(x)
        x = self.apply_output_activation(x)
        return x

