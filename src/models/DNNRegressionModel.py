# === Standard imports ===
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

# === External libraries ===
from omegaconf import DictConfig
from loguru import logger
import lightning as L
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score
from torch.optim.lr_scheduler import ReduceLROnPlateau

# === Local utility imports ===
from src.utils.Plotting import plot_metrics, compute_feature_importance_and_correlation_plot
from src.utils.LossFunctions import WeightedTailLoss, QuantileAwareLoss, InverseGaussianWeightedLoss, build_loss_function

import torchmetrics
from torchmetrics.regression import MeanSquaredLogError
from src.models.ModelMetrics import MeanFractionalBias, MedianAbsoluteError, RootMeanSquaredLogError

from src.models.ModelArchitectures import SimpleDNN

class VBFDNNRegression(L.LightningModule):
    def __init__(self, config_object : DictConfig):
        super().__init__()

        self.dropout_prob = config_object.train.dropout_probability
        self.NN_layers = config_object.model.layers
    
        self.learning_rate = config_object.train.learning_rate
        self.weight_decay = config_object.train.weight_decay

        self.loss_name = config_object.train.loss_fn
        self.loss_params = config_object.train.get('loss_fn_params', {})
        self.loss_fn = None 
        # This needs to be set up after the data module because of complicated functions like quantile loss which need y_train data
        # Note that doing this before would cause data leakage

        self.lr_scheduler_patience = config_object.train.lr_scheduler_patience
        
        # Keep track of losses
        self.train_losses = []
        self.val_losses = []
   
        # Metrics
        self.train_losses, self.val_losses = [], []
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()
        self.r2 = torchmetrics.R2Score()
        self.mfb = MeanFractionalBias()
        self.median_ae = MedianAbsoluteError()
        self.rmsle = RootMeanSquaredLogError()

        # Results
        self.result_dir = 'results/' + config_object.model.name + '/'
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
        
        input_dim = dm.train_dataset.tensors[0].shape[1]
        
        self.model = SimpleDNN(input_dim, self.NN_layers, self.dropout_prob, output_activation='softplus')

        # Loss function set up now as it can depend on the quantiles, hence on the training data
        self.loss_fn = build_loss_function(self.loss_name, self.loss_params, dm.quantiles) 
            
        logger.info(f"Model input dimension: {input_dim}")

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze()
        loss = self.loss_fn(y_hat, y)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # Lightning automatically aggregates epoch metrics, so we just grab it
        epoch_loss = self.trainer.callback_metrics["train_loss"].item()
        self.train_losses.append(epoch_loss)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze()
        loss = self.loss_fn(y_hat, y)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        epoch_loss = self.trainer.callback_metrics["val_loss"].item()
        self.val_losses.append(epoch_loss)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return {"labels" : y, "predictions" : self.model(x)}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze()
        for metric in [self.mae, self.mse, self.median_ae, self.mfb, self.rmsle, self.r2]:
            metric.update(y_hat, y)

    def on_test_epoch_end(self):
        # Manually run prediction over test set
        loader = self.trainer.datamodule.test_dataloader()
        y_true_list, y_pred_list = [], []
        for batch in loader:
            batch = self.transfer_batch_to_device(batch, self.device, 0)
            res = self.predict_step(batch, batch_idx=0)   # use predict_step to get {"labels","predictions"}
            y_true_list.append(res["labels"].cpu())
            y_pred_list.append(res["predictions"].cpu())
            
        y_true_test = torch.cat(y_true_list).cpu().detach().numpy().squeeze()
        y_pred_test = torch.cat(y_pred_list).cpu().detach().numpy().squeeze()
        
        train_losses_arr = np.array(self.train_losses)
        val_losses_arr = np.array(self.val_losses)

        y_true_test_tensor = torch.tensor(y_true_test, dtype=torch.float32, device=self.device)

        plot_metrics(train_losses_arr, val_losses_arr, y_true_test, y_pred_test, self.loss_fn, folder_name=self.result_dir)
    
        compute_feature_importance_and_correlation_plot(
            model=self.model, datamodule=self.trainer.datamodule, device=self.device,
            result_dir=self.result_dir, feature_names=self.trainer.datamodule.pretty_feature_names, trainer=self.trainer
        )

        self.log('Mean Absolute Error', self.mae.compute())
        self.log('Mean Squared Error', self.mse.compute())
        self.log('R-squared coefficient', self.r2.compute())
        self.log('Mean Fractional Bias', self.mfb.compute())
        self.log('Meadia_AE', self.median_ae.compute())
        self.log('Root Mean Squared Log Error', self.rmsle.compute())
        
        self.train_losses.clear()
        self.val_losses.clear()
        
        for metric in [self.mae, self.mse, self.r2, self.mfb, self.median_ae, self.rmsle , self.r2]:
            metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.lr_scheduler_patience)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',   # call scheduler.step() every epoch
                'monitor': 'val_loss', # metric to monitor for ReduceLROnPlateau
            }
        }