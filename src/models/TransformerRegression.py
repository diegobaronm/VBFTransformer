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
import random 

# === External libraries ===
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import lightning as L
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.models.ModelArchitectures import BasicTransformer, ConcatenationTransformer, ExtraFeatureTransformer
from ptflops import get_model_complexity_info
from torchinfo import summary  # or from torchsummary import summary

# === Local utility imports ===
from src.utils.Plotting import plot_metrics, plot_and_save_attention

from src.utils.LossFunctions import WeightedTailLoss, QuantileAwareLoss, InverseGaussianWeightedLoss, build_loss_function


from torchmetrics.regression import MeanSquaredLogError
from ignite.metrics.regression import MedianAbsoluteError

from torchmetrics import Metric

class FractionalBias(Metric):
    full_state_update = False
    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        eps = 1e-6
        self.sum += torch.sum(2.0 * (preds - target) / (preds + target + eps))
        self.total += target.numel()

    def compute(self):
        return self.sum / self.total
        

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

def compute_attention_sparsity(attn_weights, threshold=0.05):
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
        
        num_node_inputs = config_object.model.n_particles * len(config_object.dataset.scaling_dict.keys()) + len(config_object.dataset.extra_scaling_dict.keys())
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
    
        # Results
        self.result_dir = 'results/' + config_object.model.name + '/'

        self.model_dictionary = config_object.get('model')
        OmegaConf.set_struct(self.model_dictionary, False)

    # This is always called after the data module is setup
    def setup(self, stage=None):
        if (stage != 'predict'):
            dm = self.trainer.datamodule
        
            # Create the config file: 
            self.model_dictionary["input_dim"] = dm.input_dim
        else: 
            self.model_dictionary["input_dim"] = 8
             
        if self.trainer.datamodule.using_cross_attention:
            self.model = ExtraFeatureTransformer(self.model_dictionary)
            logger.info(f"Using extra feature Tranformer")
        
        elif self.trainer.datamodule.compute_pairing_tokens:
            self.model_dictionary["event_dim"] = dm.event_dim
            self.model = ConcatenationTransformer(self.model_dictionary)
            logger.info(f"Particle input dim: {dm.input_dim}, interaction input dim: {dm.event_dim}")
            
        else:
            self.model = BasicTransformer(self.model_dictionary)
        
        self.model = self.model.to(self.device)
        self.model.compile()

        # Loss function set up now as it can depend on the quantiles, hence on the training data
        if (stage != 'predict'):
            self.loss_fn = build_loss_function(self.loss_name, self.loss_params, dm.quantiles) 

        # Reset Timer each time we start training
        self.train_start_time = time.time()
        
    def training_step(self, batch, batch_idx):

        if self.trainer.datamodule.using_cross_attention or self.trainer.datamodule.compute_pairing_tokens:
            (particles, metadata, y) = batch
            y_hat = self.model(particles, metadata).squeeze()
        else:
            x, y = batch
            y_hat = self.model(x).squeeze()
            
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.train_losses.append(loss.detach().cpu().item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.trainer.datamodule.using_cross_attention or self.trainer.datamodule.compute_pairing_tokens:
            (particles, metadata, y) = batch
            y_hat = self.model(particles, metadata).squeeze()
        else:
            x, y = batch
            y_hat = self.model(x).squeeze()
            
        loss = self.loss_fn(y_hat, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_losses.append(loss.detach().cpu().item())
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.trainer.datamodule.using_cross_attention or self.trainer.datamodule.compute_pairing_tokens:
            (particles, metadata, y) = batch
            preds, attn_weights = self.model(particles, metadata, return_attentions=True)
            inputs = particles
        else:
            x, y = batch
            preds, attn_weights = self.model(x, return_attentions=True)
            inputs = x


        target_scaler = self.trainer.datamodule.target_scaler

        # Move to CPU and reshape for sklearn
        y_cpu = y.cpu().numpy()
        preds_cpu = preds.cpu().numpy()
        
        # Ensure 2D shape for sklearn scaler
        if y_cpu.ndim == 1:
            y_cpu = y_cpu.reshape(-1, 1)
        if preds_cpu.ndim == 1:
            preds_cpu = preds_cpu.reshape(-1, 1)
    
        return {
            "inputs": inputs, 
            "labels": target_scaler.inverse_transform(y_cpu),  # Move to CPU
            "predictions": target_scaler.inverse_transform(preds_cpu),  # Move to CPU
            "attentions": attn_weights
        }
    
    def test_step(self, batch, batch_idx):
        if self.trainer.datamodule.using_cross_attention or self.trainer.datamodule.compute_pairing_tokens:
            (particles, metadata, y) = batch
            y_hat = self.model(particles, metadata).squeeze()
        else:
            x, y = batch
            y_hat = self.model(x).squeeze()
   
        for metric in [self.test_mae, self.test_mse, self.test_r2, self.test_medae, self.test_rmsle, self.test_mfb]:
            metric.update(y_hat, y)

    def on_test_epoch_end(self):

        self.model.eval()
        training_duration_sec = time.time() - self.train_start_time
        training_duration_hms = str(timedelta(seconds=int(training_duration_sec)))

        # Manually run prediction over test set
        loader = self.trainer.datamodule.test_dataloader()
        target_scaler = self.trainer.datamodule.target_scaler

        y_true_list, y_pred_list, attn_list = [], [], []
        first_inputs = None
        
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

            if first_inputs is None:
                first_inputs = res["inputs"]
    
        entropies = []
        sparsities = []
        diversities = []

        # Flatten batches and layers
        # attn_list: list of [num_layers x batch x num_heads x seq x seq]
        # transpose to [num_layers][batch x heads x seq x seq]
        layerwise = list(zip(*attn_list))  # each element is list over batches

        # Needs to be done to avoid taking too long for huge batches
        sample_ratio = 0.05 # take 20% of batches, adjust as needed
        sample_size = max(1, int(len(layerwise) * sample_ratio))
        
        for layer_attn_batches in layerwise:
            # randomly sample some batches from this layer
            sampled_batches = random.sample(layer_attn_batches, sample_size)
        
            # stack over batches: shape [sample_size, heads, seq, seq]
            attn_tensor = torch.cat(sampled_batches, dim=0)
        
            entropy = compute_attention_entropy(attn_tensor)
            sparsity = compute_attention_sparsity(attn_tensor)
            diversity = compute_head_diversity(attn_tensor)
        
            entropies.append(entropy)
            sparsities.append(sparsity)
            diversities.append(diversity)

        # Recale back the targets
        
        
        
        y_true_list = [torch.tensor(y) if isinstance(y, np.ndarray) else y for y in y_true_list]
        y_true_test = torch.cat(y_true_list).cpu().detach().numpy().squeeze()

        y_pred_list = [torch.tensor(y) if isinstance(y, np.ndarray) else y for y in y_pred_list]
        y_pred_test = torch.cat(y_pred_list).cpu().detach().numpy().squeeze()
        # Save to csv: 
        
        # If your predictions/labels are multi-dimensional, you may need to flatten or create column names
        # For simplicity, assume 1D
        df = pd.DataFrame({
            "y_true": y_true_test,
            "y_pred": y_pred_test
        })
        
        # Save to CSV
        df.to_csv("Training_predictions.csv", index=False)

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
        y_pred_test_tensor = torch.tensor(y_pred_test, dtype=torch.float32, device=self.device)
        
        loss_human1 = self.loss_fn(torch.tensor(self.trainer.datamodule.M_reco_human, dtype=torch.float32, device=self.device),y_true_test_tensor).item()
        loss_human2 = self.loss_fn(torch.tensor(self.trainer.datamodule.M_mmc_human, dtype=torch.float32, device=self.device), y_true_test_tensor).item()

        plot_metrics(avg_train_loss, avg_val_loss, y_true_test, y_pred_test, -1,-1, self.loss_fn, folder_name=self.result_dir)

        plot_and_save_attention(
            attn_per_example=attn_list[0],
            save_dir=self.result_dir,
        )
        
        # Now compute the metrics (disable distributed sync since you're likely running on single GPU/CPU)
        mae_val = mean_absolute_error(y_true_test, y_pred_test)
        mse_val = mean_squared_error(y_true_test, y_pred_test)
        r2_val  = r2_score(y_true_test, y_pred_test)
        mfb = mean_fractional_bias(y_pred_test, y_true_test)
        median_abs_error_val = np.median(np.abs(y_true_test - y_pred_test))

        rmsle = root_mean_squared_log_error(y_pred_test, y_true_test)
   
        param_count = count_parameters(self.model)
        
        mem_usage = get_memory_usage()
            
        all_metrics = {
            "Mean Absolute Error (MAE)": float(mae_val),
            "Mean Squared Error (MSE)": float(mse_val),
            "R² Score": float(r2_val),
            "Mean Fractional Bias (MFB)": float(mfb),
            "Root Mean Squared Log Error (RMSLE)": float(rmsle),
            "Median Absolute Error": float(median_abs_error_val),
            "Number of Parameters": int(param_count),
            "Peak GPU Memory Usage (MB)": float(mem_usage),
            "Total Training Time (HH:MM:SS)": str(training_duration_hms),
            "Training Batch Size": self.trainer.datamodule.train_batch_size,
            "Validation Batch Size": self.trainer.datamodule.val_batch_size,
            "Number of Training Events": len(self.trainer.datamodule.train_dataset),
        }

        # Log and store attention metrics in the report dictionary
        for i, (e, s, d) in enumerate(zip(entropies, sparsities, diversities)):
            print(f"Layer {i}: Entropy={e:.4f}, Sparsity={s:.4f}, Diversity={d:.4f}")
        
            all_metrics[f"Layer {i} - Attention Entropy"]  = round(float(e), 4)
            all_metrics[f"Layer {i} - Attention Sparsity"] = round(float(s), 4)
            all_metrics[f"Layer {i} - Head Diversity"]     = round(float(d), 4)
        
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
            "memory_MB": float(mem_usage),
            "Training Duration": str(training_duration_hms),
            "Training Batch Size": str(self.trainer.datamodule.train_batch_size), 
            "Validation Batch Size": str(self.trainer.datamodule.val_batch_size), 
            "Event Count": str(len(self.trainer.datamodule.train_dataset)),   
        }

        with open(os.path.join(self.result_dir, "model_metrics.yaml"), "w") as f:
            OmegaConf.save(config=OmegaConf.create(all_metrics), f=f.name)

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
