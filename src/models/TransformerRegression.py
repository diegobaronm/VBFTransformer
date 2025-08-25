 # === Standard imports ===
import os
import io
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random 

# === External libraries ===
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import lightning as L
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.models.ModelArchitectures import BasicTransformer, ConcatenationTransformer
from ptflops import get_model_complexity_info
from torchinfo import summary  # or from torchsummary import summary
import torchmetrics 

# === Local utility imports ===
from src.utils.Plotting import plot_metrics, plot_and_save_attention
from src.utils.LossFunctions import WeightedTailLoss, QuantileAwareLoss, InverseGaussianWeightedLoss, build_loss_function
from torchmetrics.regression import MeanSquaredLogError
from src.models.ModelMetrics import AttentionEntropy, AttentionSparsity, HeadDiversity # transformer specific metrics
from src.models.ModelMetrics import MeanFractionalBias, MedianAbsoluteError, RootMeanSquaredLogError # general metrics
    
class VBFTransformerRegression(L.LightningModule):
    def __init__(self, config_object : DictConfig):
        super().__init__()

        self.dropout_prob = config_object.train.dropout_probability
        self.input_embedder_NN_layers = config_object.model.input_embedder_NN_layers
        self.num_layers = config_object.model.num_layers
        self.num_heads = config_object.model.num_heads
        self.output_activation = config_object.model.output_activation
        self.pooling_type = config_object.model.pooling_type
    
        self.learning_rate = config_object.train.learning_rate
        self.weight_decay = config_object.train.weight_decay
        self.lr_scheduler_patience = config_object.train.lr_scheduler_patience

        self.loss_name = config_object.train.loss_fn
        self.loss_params = config_object.train.get('loss_fn_params', {})
        # Loss criterion is initialized in setup as we need datamodule to be initialized
    
        self.result_dir = 'results/' + config_object.model.name + '/'

        self.model_dictionary = config_object.get('model')
        OmegaConf.set_struct(self.model_dictionary, False) # False allows the dictionary object to be changed

        # Metrics computed during training/eval/test
        self.train_losses, self.val_losses = [], []
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()
        self.r2 = torchmetrics.R2Score()
        self.mfb = MeanFractionalBias()
        self.median_ae = MedianAbsoluteError()
        self.rmsle = RootMeanSquaredLogError()
        self.attn_entropy = AttentionEntropy()
        self.attn_sparsity = AttentionSparsity(threshold=0.05)
        self.head_diversity = HeadDiversity()

        # Private variable to ensure set up is not done twice when calling training and test scripts back to back
        self._has_setup = False

    # This is always called after the data module is setup
    def setup(self, stage=None, datamodule=None):
        if self._has_setup:
            return
        self._has_setup = True
        
        if datamodule is None:
            dm = self.trainer.datamodule
        else:
            dm = datamodule
        
        # Create the config file: 
        self.model_dictionary["input_dim"] = dm.input_dim

        if dm.compute_pairing_tokens:
            self.model_dictionary["event_dim"] = dm.event_dim
            self.model = ConcatenationTransformer(self.model_dictionary)
            logger.info(f"Particle input dim: {dm.input_dim}, interaction input dim: {dm.event_dim}")
            
        else:
            self.model = BasicTransformer(self.model_dictionary)
        
        self.model = self.model.to(self.device)
        self.model.compile()

        # Loss function set up now as it can depend on the quantiles, hence on the training data
        self.loss_fn = build_loss_function(self.loss_name, self.loss_params, dm.quantiles) 
        
    def training_step(self, batch, batch_idx):

        if self.trainer.datamodule.compute_pairing_tokens:
            (particles, metadata, y) = batch
            y_hat = self.model(particles, metadata).squeeze()
        else:
            x, y = batch
            y_hat = self.model(x).squeeze()
            
        loss = self.loss_fn(y_hat, y)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        # Lightning automatically aggregates epoch metrics, so we just grab it
        epoch_loss = self.trainer.callback_metrics["train_loss"].item()
        self.train_losses.append(epoch_loss)
    
    def validation_step(self, batch, batch_idx):
        if self.trainer.datamodule.compute_pairing_tokens:
            (particles, metadata, y) = batch
            y_hat = self.model(particles, metadata).squeeze()
        else:
            x, y = batch
            y_hat = self.model(x).squeeze()
            
        loss = self.loss_fn(y_hat, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def on_validation_epoch_end(self):
        # Lightning automatically aggregates epoch metrics, so we just grab it
        epoch_loss = self.trainer.callback_metrics["val_loss"].item()
        self.val_losses.append(epoch_loss)
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.trainer.datamodule.compute_pairing_tokens:
            (particles, metadata, y) = batch
            preds, attn_weights = self.model(particles, metadata, return_attentions=True)
        else:
            x, y = batch
            preds, attn_weights = self.model(x, return_attentions=True)

        target_scaler = self.trainer.datamodule.target_scaler

        # Move to CPU and reshape for sklearn
        y_cpu = y.cpu()
        preds_cpu = preds.cpu()
        
        # Ensure 2D shape for sklearn scaler
        if y_cpu.ndim == 1:
            y_cpu = y_cpu.reshape(-1, 1)
        if preds_cpu.ndim == 1:
            preds_cpu = preds_cpu.reshape(-1, 1)
        
        return { 
            "labels": target_scaler.inverse_transform(y_cpu),  
            "predictions": target_scaler.inverse_transform(preds_cpu), 
            "attentions": attn_weights
        }
    
    def test_step(self, batch, batch_idx):
        if self.trainer.datamodule.compute_pairing_tokens:
            (particles, metadata, y) = batch
            y_hat = self.model(particles, metadata).squeeze()
        else:
            x, y = batch
            y_hat = self.model(x).squeeze()
   
        for metric in [self.mae, self.mse, self.median_ae, self.mfb, self.rmsle, self.r2]:
            metric.update(y_hat, y)

    
    def on_test_epoch_end(self):
        self.model.eval()
        
        # Manually run prediction over test set
        loader = self.trainer.datamodule.test_dataloader()
        target_scaler = self.trainer.datamodule.target_scaler

        y_true_list, y_pred_list, attn_list = [], [], []
        
        for batch in loader:
            batch = self.transfer_batch_to_device(batch, self.device, 0)

            with torch.no_grad():
                res = self.predict_step(batch, batch_idx=0)
           
            y_true_list.append(res["labels"])
            y_pred_list.append(res["predictions"])

            if "attentions" in res:
                # List of tensors, one per layer: [batch, seq_len, seq_len]
                attn_list.append([a.cpu() for a in res["attentions"]])
                torch.cuda.empty_cache() # Need to clean when num events is large

        # Flatten batches and layers
        # attn_list: list of [num_layers x batch x num_heads x seq x seq]
        # transpose to [num_layers][batch x heads x seq x seq]
        layerwise = list(zip(*attn_list))  # each element is list over batches

        # Needs to be done to avoid taking too long for huge batches
        sample_ratio = 0.05 # take 5% of batches, adjust as needed
        sample_size = max(1, int(len(layerwise) * sample_ratio))
        
        for index, layer_attn_batches in enumerate(layerwise):
            # randomly sample some batches from this layer
            sampled_batches = random.sample(layer_attn_batches, sample_size)
        
            # stack over batches: shape [sample_size, heads, seq, seq]
            attn_tensor = torch.cat(sampled_batches, dim=0)
            
            self.log(f'Attention/Entropy/layer_{index}', self.attn_entropy(attn_tensor) )
            self.log(f'Attention/Sparsity/layer_{index}', self.attn_sparsity(attn_tensor) )
            self.log(f'Attention/Diverisity/layer_{index}', self.head_diversity(attn_tensor) )
            
        y_true_list = [torch.tensor(y) if isinstance(y, np.ndarray) else y for y in y_true_list]
        y_true_test = torch.cat(y_true_list).cpu().detach().numpy().squeeze()

        y_pred_list = [torch.tensor(y) if isinstance(y, np.ndarray) else y for y in y_pred_list]
        y_pred_test = torch.cat(y_pred_list).cpu().detach().numpy().squeeze()

        train_losses_arr = np.array(self.train_losses)
        val_losses_arr = np.array(self.val_losses)

        y_true_test_tensor = torch.tensor(y_true_test, dtype=torch.float32, device=self.device)
        y_pred_test_tensor = torch.tensor(y_pred_test, dtype=torch.float32, device=self.device)

        self.log('Mean Absolute Error', self.mae.compute())
        self.log('Mean Squared Error', self.mse.compute())
        self.log('R-squared coefficient', self.r2.compute())
        self.log('Mean Fractional Bias', self.mfb.compute())
        self.log('Meadia_AE', self.median_ae.compute())
        self.log('Root Mean Squared Log Error', self.rmsle.compute())
    
        plot_metrics(train_losses_arr, val_losses_arr, y_true_test, y_pred_test, self.loss_fn, folder_name=self.result_dir)
        plot_and_save_attention(attn_per_example=attn_list[0], save_dir=self.result_dir)
   
        # Clear lists for next test
        self.train_losses.clear()
        self.val_losses.clear()
        
        for metric in [self.mae, self.mse, self.r2, self.mfb, self.median_ae, self.rmsle , self.r2, self.attn_entropy, self.attn_sparsity,  self.head_diversity]:
            metric.reset()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.lr_scheduler_patience)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',   # call scheduler.step() every epoch
                'monitor': 'val_loss', # metric to monitor for changing Learning Rate
            }
        }
