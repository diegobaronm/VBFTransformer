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
sys.path.append(os.path.abspath("src/utils"))
from Plotting import plot_metrics, permutation_feature_importance, compute_feature_importance_and_correlation_plot
from LossFunctions import WeightedTailLoss, QuantileAwareLoss, InverseGaussianWeightedLoss, build_loss_function


class BasicTransformer(nn.Module):
    def __init__(self, input_dim, nhead, num_layers):
        super(BasicTransformer, self).__init__()
        
        # Input embedding head
        self.input_embedder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
          #  nn.Dropout(0.1),
            nn.Linear(32, 64),
        )
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=nhead,batch_first=True),
            num_layers=num_layers, enable_nested_tensor=False)

        self.token_type_embeddings = nn.Embedding(3, 64)  # 3 tokens, embedding dim=64
        
        self.output_classifier_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
    def mean_pooling(self, x):
        return x.mean(dim=1)

    def forward(self, x):
        x = self.input_embedder(x)  # (batch_size, 3, 64)
        
        batch_size = x.size(0)
        device = x.device
        
        token_type_ids = torch.tensor([0, 1, 2], device=device).unsqueeze(0).repeat(batch_size, 1)
        token_type_embeds = self.token_type_embeddings(token_type_ids)  # (batch_size, 3, 64)
        
        x = x + token_type_embeds
        
        x = self.transformer_encoder(x)
        x = self.mean_pooling(x)
        x = self.output_classifier_head(x)
        return x