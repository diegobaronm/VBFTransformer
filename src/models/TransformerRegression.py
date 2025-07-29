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
from src.models.ModelArchitectures import BasicTransformer

# === Local utility imports ===
sys.path.append(os.path.abspath("src/utils"))
from Plotting import plot_metrics, permutation_feature_importance, compute_feature_importance_and_correlation_plot
from LossFunctions import WeightedTailLoss, QuantileAwareLoss, InverseGaussianWeightedLoss, build_loss_function

class VBFTransformerRegression(L.LightningModule):
    def __init__(self, config_object : DictConfig):
        super().__init__()

        self.dropout_prob = config_object.train.dropout_probability
        self.NN_layers = config_object.model.layers

        self.num_layers = config_object.model.num_layers
        self.num_heads = config_object.model.num_heads

        self.input_dim = 5 # The dimensionality of each token
        
        num_node_inputs = config_object.model.n_particles * len(config_object.model.features) + len(config_object.model.extra_features)
        logger.info(f"Number of node inputs in the model {num_node_inputs}")
    
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
        self.test_mae = MeanAbsoluteError()
        self.test_mse = MeanSquaredError()
        self.test_r2 = R2Score()

        # Results
        self.result_dir = 'results/' + config_object.model.name + '/'

    # This is always called after the data module is setup
    def setup(self, stage=None):
        dm = self.trainer.datamodule
        
        self.model = BasicTransformer(self.input_dim, nhead=self.num_heads, num_layers=self.num_layers)
        self.model = self.model.to(self.device)
        self.model.compile()

        # Loss function set up now as it can depend on the quantiles, hence on the training data
        self.loss_fn = build_loss_function(self.loss_name, self.loss_params, dm.quantiles) 

    def training_step(self, batch, batch_idx):
        x, y = batch
    
        y_hat = self.model(x).squeeze()
        loss = self.loss_fn(y_hat, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=True)

        self.train_losses.append(loss.detach().cpu().item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze()
        loss = self.loss_fn(y_hat, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_losses.append(loss.detach().cpu().item())
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return {"labels" : y, "predictions" : self.model(x)}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze()
        self.test_mae.update(y_hat, y)
        self.test_mse.update(y_hat, y)
        self.test_r2.update(y_hat, y)

    def on_test_epoch_end(self):
        # Manually run prediction over test set
        loader = self.trainer.datamodule.test_dataloader()
        target_scaler = self.trainer.datamodule.target_scaler

        
        y_true_list, y_pred_list = [], []
        for batch in loader:
            batch = self.transfer_batch_to_device(batch, self.device, 0)
            res = self.predict_step(batch, batch_idx=0)   # use predict_step to get {"labels","predictions"}
            y_true_list.append(res["labels"].cpu())
            y_pred_list.append(res["predictions"].cpu())

        # Recale back the targets
        y_true_test = torch.cat(y_true_list).cpu().detach().numpy().squeeze()
        y_pred_test = torch.cat(y_pred_list).cpu().detach().numpy().squeeze()
        y_pred_test = target_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
        y_true_test = target_scaler.inverse_transform(y_true_test.reshape(-1, 1)).flatten()

        train_batch_num = len(self.trainer.datamodule.train_dataloader())
        val_batch_num = len(self.trainer.datamodule.val_dataloader())
        train_losses_arr = np.array(self.train_losses)
        val_losses_arr = np.array(self.val_losses)
        num_epochs = int(len(self.train_losses) / train_batch_num)
        
        # We can reshape only because we have drop=True on both of them
        # Calculate number of epochs based on train losses
        num_epochs_train = len(train_losses_arr) // train_batch_num
        num_epochs_val = len(val_losses_arr) // val_batch_num
        
        # Trim train_losses and val_losses to exact multiples
        train_losses_arr = train_losses_arr[:num_epochs_train * train_batch_num]
        val_losses_arr = val_losses_arr[:num_epochs_val * val_batch_num]
        
        # Now reshape safely
        avg_train_loss = train_losses_arr.reshape(num_epochs_train, train_batch_num).mean(axis=1)
        avg_val_loss = val_losses_arr.reshape(num_epochs_val, val_batch_num).mean(axis=1)

        y_true_test_tensor = torch.tensor(y_true_test, dtype=torch.float32, device=self.device)
        
        loss_human1 = self.loss_fn(torch.tensor(self.trainer.datamodule.M_reco_human, dtype=torch.float32, device=self.device),y_true_test_tensor).item()
        loss_human2 = self.loss_fn(torch.tensor(self.trainer.datamodule.M_mmc_human, dtype=torch.float32, device=self.device), y_true_test_tensor).item()

        plot_metrics(avg_train_loss, avg_val_loss, y_true_test, y_pred_test, -1,-1, self.loss_fn, folder_name=self.result_dir)

        # Dont plot correlations for Transformer
        ''' 
        compute_feature_importance_and_correlation_plot(
            model=self.model, datamodule=self.trainer.datamodule, device=self.device,
            result_dir=self.result_dir, feature_names=self.trainer.datamodule.pretty_feature_names, trainer=self.trainer
        )
        '''
        
        # Reset metrics
        self.test_mae.reset()
        self.test_mse.reset()
        self.test_r2.reset()

        # Clear lists for next test
        self.train_losses.clear()
        self.val_losses.clear()


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