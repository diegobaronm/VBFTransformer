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
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import timedelta

# === External libraries ===
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import lightning as L
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.models.ModelArchitectures import BasicTransformer
from ptflops import get_model_complexity_info
from torchinfo import summary  # or from torchsummary import summary

# === Local utility imports ===
from src.utils.Plotting import plot_metrics, permutation_feature_importance, compute_feature_importance_and_correlation_plot
from src.utils.LossFunctions import WeightedTailLoss, QuantileAwareLoss, InverseGaussianWeightedLoss, build_loss_function


def mean_fractional_bias(y_pred, y_true):
    """Compute Mean Fractional Bias (MFB)"""
    eps = 1e-6  # To avoid division by zero
    return 2.0 * np.mean((y_pred - y_true) / (y_pred + y_true + eps))

def root_mean_squared_log_error(y_pred, y_true):
    """Compute Root Mean Squared Logarithmic Error (RMSLE)"""
    eps = 1e-6  # Avoid log(0)
    return np.sqrt(np.mean((np.log1p(y_pred + eps) - np.log1p(y_true + eps))**2))

def compute_attention_entropy(attn_weights):
    """
    attn_weights: Tensor of shape [batch_size, num_heads, seq_len, seq_len]
    """
    eps = 1e-9
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=-1)  # [batch, heads, seq_len]
    avg_entropy = entropy.mean().item()
    return avg_entropy

def compute_attention_sparsity(attn_weights, threshold=0.1):
    """
    Proportion of attention values greater than `threshold`
    """
    sparse_mask = attn_weights > threshold
    sparsity_ratio = sparse_mask.sum().item() / attn_weights.numel()
    return sparsity_ratio

def compute_head_diversity(attn_weights):
    """
    Measures how different attention heads are
    Cosine similarity between flattened attention maps of heads
    """
    batch_size, num_heads, seq_len, _ = attn_weights.shape
    diversity_scores = []
    for i in range(batch_size):
        for j in range(num_heads):
            for k in range(j+1, num_heads):
                a = attn_weights[i, j].flatten()
                b = attn_weights[i, k].flatten()
                sim = F.cosine_similarity(a, b, dim=0).item()
                diversity_scores.append(1 - sim)
    return np.mean(diversity_scores)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_flops(model, input_shape):
    macs, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=False)
    return macs, params

def get_memory_usage():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6  # In MB

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
        
    
class VBFTransformerRegression(L.LightningModule):
    def __init__(self, config_object : DictConfig):
        super().__init__()

        self.dropout_prob = config_object.train.dropout_probability
        self.input_embedder_NN_layers = config_object.model.input_embedder_NN_layers
        self.num_layers = config_object.model.num_layers
        self.num_heads = config_object.model.num_heads
        self.output_activation = config_object.model.output_activation
        self.pooling_type = config_object.model.pooling_type
        
        num_node_inputs = config_object.model.n_particles * len(config_object.model.features) + len(config_object.model.extra_features)
        logger.info(f"Number of node inputs in the model {num_node_inputs}")

        logger.info(f"Output Activation Function {self.output_activation}")
    
        self.learning_rate = config_object.train.learning_rate
        self.weight_decay = config_object.train.weight_decay

        self.loss_name = config_object.train.loss_fn
        self.loss_params = config_object.train.get('loss_fn_params', {})
        self.loss_fn = None 
        # This needs to be set up after the data module because of complicated functions like quantile loss which need y_train data
        # Note that doing this before would cause data leakage
    
        self.lr_scheduler_patience = config_object.train.lr_scheduler_patience

        # Metrics computed during training/eval/test
        self.train_losses, self.val_losses = [], []

        self.test_mae = MeanAbsoluteError()
        self.test_mse = MeanSquaredError()
        self.test_r2 = R2Score()

        # Results
        self.result_dir = 'results/' + config_object.model.name + '/'

        self.model_dictionary = config_object.get('model')
        OmegaConf.set_struct(self.model_dictionary, False)

    # This is always called after the data module is setup
    def setup(self, stage=None):
        dm = self.trainer.datamodule

        # Create the config file:

        self.model_dictionary["input_dim"] = dm.input_dim
        self.model_dictionary["n_tokens"] = dm.n_particles
        

            # "norm_type": "layernorm",     # or "batchnorm" or None
    
        
        self.model = BasicTransformer(self.model_dictionary)
        
        self.model = self.model.to(self.device)
        self.model.compile()

        # Loss function set up now as it can depend on the quantiles, hence on the training data
        self.loss_fn = build_loss_function(self.loss_name, self.loss_params, dm.quantiles) 

        # Reset Timer each time we start training
        self.train_start_time = time.time()
        
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
        model.eval()
        training_duration_sec = time.time() - self.train_start_time
        training_duration_hms = str(timedelta(seconds=int(training_duration_sec)))

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
        
        compute_feature_importance_and_correlation_plot(
            model=self.model, datamodule=self.trainer.datamodule, device=self.device,
            result_dir=self.result_dir, feature_names=self.trainer.datamodule.pretty_feature_names, trainer=self.trainer, transformer=True
        )

        
        mae_val = mean_absolute_error(y_true_test, y_pred_test)
        mse_val = mean_squared_error(y_true_test, y_pred_test)
        r2_val  = r2_score(y_true_test, y_pred_test)
        mfb = mean_fractional_bias(y_pred_test, y_true_test)
        median_abs_error_val = np.median(np.abs(y_true_test - y_pred_test))

   
        rmsle = root_mean_squared_log_error(y_pred_test, y_true_test)
    

        param_count = count_parameters(self.model)
    
    
        macs, _ = compute_flops(
        self.model,       input_shape=(self.trainer.datamodule.n_particles, self.trainer.datamodule.input_dim)
        )
    

        mem_usage = get_memory_usage()
    
        # Capture the summary output as a string
        buffer = io.StringIO()
        sys.stdout = buffer
        summary(self.model, input_size=(1, self.trainer.datamodule.n_particles, self.trainer.datamodule.input_dim))
        sys.stdout = sys.__stdout__  # reset stdout
        
        model_summary_str = buffer.getvalue()
        
        all_metrics = {
            "Mean Absolute Error (MAE)": float(mae_val),
            "Mean Squared Error (MSE)": float(mse_val),
            "R² Score": float(r2_val),
            "Mean Fractional Bias (MFB)": float(mfb),
            "Root Mean Squared Log Error (RMSLE)": float(rmsle),
            "Median Absolute Error": float(median_abs_error_val),
            "Number of Parameters": int(param_count),
            "Estimated FLOPs (MFLOPs)": round(float(macs.split()[0]) * 2, 2),
            "Peak GPU Memory Usage (MB)": float(mem_usage),
            "Total Training Time (HH:MM:SS)": str(training_duration_hms),
            "Training Batch Size": self.trainer.datamodule.train_batch_size,
            "Validation Batch Size": self.trainer.datamodule.val_batch_size,
            "Number of Training Events": len(self.trainer.datamodule.train_dataset),
        }
        
        # Format for output
        metrics_str = "\n".join([f"{k:<40}: {v}" for k, v in sorted(all_metrics.items())])
        
        # Write report to file
        report_path = os.path.join(self.result_dir, "model_report.txt")
        with open(report_path, 'w') as f:
            f.write("#" * 70 + "\n")
            f.write("#                 MODEL TRAINING REPORT                #\n")
            f.write("#" * 70 + "\n\n")
        
            f.write("## === Training Metrics === ##\n")
            f.write(metrics_str + "\n\n")
        
            f.write("## === Model Summary === ##\n")
            f.write(model_summary_str + "\n")
        
            f.write("#" * 70 + "\n")
            f.write("#                 END OF REPORT                        #\n")
            f.write("#" * 70 + "\n")       

            
        all_metrics = {
            "mae": float(mae_val),
            "mse": float(mse_val),
            "r2": float(r2_val),
            "mfb": float(mfb),
            "rmsle": float(rmsle),
            "medae": float(median_abs_error_val),
            "param_count": int(param_count),
            "Mflops": str(float(macs.split()[0]) * 2), 
            "memory_MB": float(mem_usage),
            "Training Duration": str(training_duration_hms),
            "Training Batch Size": str(self.trainer.datamodule.train_batch_size), 
            "Validation Batch Size": str(self.trainer.datamodule.val_batch_size), 
            "Event Count": str(len(self.trainer.datamodule.train_dataset)),   
        }


        
        with open(os.path.join(self.result_dir, "model_metrics.yaml"), "w") as f:
            OmegaConf.save(config=OmegaConf.create(all_metrics), f=f.name)

                
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
                'monitor': 'val_loss', # metric to monitor for the scheduler
            }
        }