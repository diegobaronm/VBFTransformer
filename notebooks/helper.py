import sys
import subprocess
from enum import Enum  
import h5py
import numpy as np
from numpy.lib import recfunctions as rfn
import pandas as pd
import numpy as np
import torch
import polars as pl
import math
import matplotlib.pyplot as plt
from numpy.lib import recfunctions as rfn
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import TensorDataset, DataLoader
import torch
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.colors as colors
import re
import json
import os
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from sklearn.metrics import r2_score

# ======================================================= #
# ===================== LOADING FILES  ================== #
# ======================================================= #

def read_h5_data(file_name):
    with h5py.File(file_name, 'r') as file:
        particles = file['INPUTS']['PARTICLES'][:, :42]  # First 43 columns/features
        metadata = file['METADATA']['EVENT_DATA'][:]

        signal_inputs = rfn.structured_to_unstructured(particles)

        particles_keys = list(file['INPUTS']['PARTICLES'].dtype.names)[:42]
        input_keys_dict = {key: idx for idx, key in enumerate(particles_keys)}

    return signal_inputs, metadata, input_keys_dict

def load_multiple_h5(files):
    all_signals = []
    all_metadata = []
    input_keys_dict = None

    for f in files:
        signals, metadata, keys_dict = read_h5_data(f)
        all_signals.append(signals)
        all_metadata.append(metadata)
        if input_keys_dict is None:
            input_keys_dict = keys_dict

    combined_signals = np.vstack(all_signals)
    combined_metadata = np.vstack(all_metadata)

    return combined_signals, combined_metadata, input_keys_dict

def flat_inputs(input_array, n_particles, feature_indices=None):
    if feature_indices is None:
        feature_indices = list(range(input_array.shape[2]))  # Use all features if none specified

    # Select specific particles and features
    output_array = input_array[:, :n_particles, feature_indices]
    
    # Transpose to (samples, features, particles)
    output_array = output_array.transpose(0, 2, 1)
    
    # Flatten to (samples, n_particles * n_features_selected)
    return output_array.reshape(-1, n_particles * len(feature_indices))

def add_features(starting_df, arr_of_features):

    data_df = starting_df

    for feature in arr_of_features:
        new_feature = feature.reshape(-1, 1)
        data_df = np.hstack((data_df, new_feature))

    print(f"The final df shape is: {data_df.shape}")
    return data_df
    
class MetadataIndex(Enum):
    # EVENT_NUMBER = 0 No need to assing this a n
    OMEGA = 1
    MZ_RECO = 2
    MZ_TRUTH = 3
    MMC_MZ = 4
    OPENING_ANGLE = 5
    LEP_MET_ANGLE_SIGNED = 6
    IS_MET_INSIDE = 7

def save_notebook_as_txt(notebook_path, output_dir, output_filename=None):
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Extract code cells
    code_cells = [
        ''.join(cell['source']) 
        for cell in nb['cells'] 
        if cell['cell_type'] == 'code'
    ]

    # Combine code and prepare save path
    code_text = '\n\n'.join(code_cells)
    os.makedirs(output_dir, exist_ok=True)

    if output_filename is None:
        output_filename = os.path.splitext(os.path.basename(notebook_path))[0] + '.txt'

    output_path = os.path.join(output_dir, output_filename)

    # Save to .txt file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(code_text)

    print(f"Code saved to: {output_path}")
    
# ======================================================= #
# ===================== SCALING CLASSES ================= #
# ======================================================= #

class LogScaler(BaseEstimator, TransformerMixin):
    def __init__(self, offset=1e-3, add_mask=False):
        self.offset = offset
        self.add_mask = add_mask

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        invalid_mask = (X + self.offset) <= 0
        log = np.log(X + self.offset)
        mask = np.isnan(log)
        if self.add_mask:
            return np.hstack([mask, log])
        return log

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        if self.add_mask:
            mask_features = [f"{feat}_mask" for feat in input_features]
            log_features = [f"log({feat})" for feat in input_features]
            return np.array(mask_features + log_features)
        else:
            return np.array([f"log({feat})" for feat in input_features])


class PhiTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        sin_phi = np.sin(X)
        cos_phi = np.cos(X)
        return np.hstack([sin_phi, cos_phi])

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        out = []
        for feat in input_features:
            out.append(f"sin({feat})")
            out.append(f"cos({feat})")
        return np.array(out)

class ArctanScaler(FunctionTransformer):
    def __init__(self):
        super().__init__(func=lambda x: np.arctan(x) * 2 / np.pi, validate=True)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        # Append or prepend something to indicate this transform was applied
        return [f"arctan_scaled_{name}" for name in input_features]


class LogMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, add_mask=False):
        self.scaler = MinMaxScaler()
        self.add_mask = add_mask

    def fit(self, X, y=None):
        X_log = np.log1p(X)
        # Fit scaler only on non-NaN values (mask out NaNs)
        self.mask_ = np.isnan(X_log)
        # Use only non-NaN values for fitting scaler (flattened)
        self.scaler.fit(X_log[~self.mask_].reshape(-1, 1))
        return self

    def transform(self, X):
        X_log = np.log1p(X)
        mask = np.isnan(X_log)

        # Prepare array to store scaled results, initially all NaN
        X_scaled = np.full_like(X_log, np.nan, dtype=float)

        # Scale only non-NaN values
        non_nan_idx = ~mask
        # MinMaxScaler expects 2D array for transform
        scaled_values = self.scaler.transform(X_log[non_nan_idx].reshape(-1, 1)).flatten()
        X_scaled[non_nan_idx] = scaled_values

        if self.add_mask:
            return np.hstack([mask, X_scaled])
        return X_scaled

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        scaled_names = [f"{name}" for name in input_features]
        if self.add_mask:
            mask_names = [f"{name}_nan_mask" for name in input_features]
            return mask_names + scaled_names
        return scaled_names


class WeightedScaler(BaseEstimator, TransformerMixin):
    def __init__(self, weights):
        # Just store the input directly without converting
        self.weights = weights
    
    def fit(self, X, y=None):
        # Convert to np.array here and validate shape
        self.weights_ = np.array(self.weights)
        print(X.shape)
        print(self.weights_.shape)
        if X.shape != self.weights_.shape:
            raise ValueError(f"Number of weights ({len(self.weights_)}) must match number of features ({X.shape[1]})")
        return self
    
    def transform(self, X):
        # Use self.weights_ here, which is guaranteed to be a numpy array
        return X / self.weights_
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(len(self.weights))]
        return [f"{name}_weighted_by_{w}" for name, w in zip(input_features, self.weights)]


class TanhScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        return 0.5 * (np.tanh((X - self.mean_) / self.std_) + 1)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return [f"Tanh{name}" for name in input_features]


# ============================================ #
# ============= Plotting Software ============ #
# ============================================ #

def plot_metrics(train_losses, val_losses, y_true_test, y_pred_test, MZ_reco_est, MMC_est, criterion, folder_name="plots"):
    os.makedirs(folder_name, exist_ok=True)
    
    # Style params
    title_fontsize = 16
    label_fontsize = 14
    legend_fontsize = 12
    tick_fontsize = 12
    linewidth = 2
    
    # === FIRST FIGURE: Loss and Mass Histogram ===
    # Residuals and Median AE
    residuals = y_pred_test - y_true_test
    median_ae = np.median(np.abs(residuals))
    
    # === FIRST FIGURE: Loss Curves and 3D Loss Landscapes ===
    fig1 = plt.figure(figsize=(18, 6))
    gs = fig1.add_gridspec(1, 3)
    ax0 = fig1.add_subplot(gs[0, 0])                       # Loss curves
    ax1 = fig1.add_subplot(gs[0, 1], projection='3d')      # 3D view 1
    ax2 = fig1.add_subplot(gs[0, 2], projection='3d')      # 3D view 2
    
    # --- Plot 1: Log Losses ---
    ax0.plot(np.log(train_losses), label='Train Loss', color='red', linewidth=linewidth)
    ax0.plot(np.log(val_losses), label='Validation Loss', color='green', linewidth=linewidth)
    
    if MZ_reco_est > 0:
        ax0.axhline(np.log(MZ_reco_est), color='blue', linestyle='--', linewidth=linewidth, label='Ln MZ-Reco Loss')
    if MMC_est > 0:
        ax0.axhline(np.log(MMC_est), color='purple', linestyle='--', linewidth=linewidth, label='Ln MMC Loss')
    
    ax0.set_xlabel('Epoch', fontsize=label_fontsize)
    ax0.set_ylabel('Log(Loss)', fontsize=label_fontsize)
    ax0.set_title('Train & Val Loss (Log Scale)', fontsize=title_fontsize)
    ax0.legend(fontsize=legend_fontsize, framealpha=0.8)
    ax0.grid(True, linestyle='--', alpha=0.5)
    ax0.tick_params(axis='both', labelsize=tick_fontsize)
    ax0.minorticks_on()
    
    # --- Compute Loss Grid ---
    preds = torch.linspace(80, 110, steps=150)
    targets = torch.linspace(80, 110, steps=150)
    P, T = torch.meshgrid(preds, targets, indexing='ij')
    loss_grid = np.zeros_like(P.numpy())
    
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            pred_tensor = torch.tensor([P[i, j].item()])
            target_tensor = torch.tensor([T[i, j].item()])
            loss_val = criterion(pred_tensor, target_tensor)
            loss_grid[i, j] = loss_val.item()
    
    X = P.numpy()
    Y = T.numpy()
    Z = loss_grid
    
    # --- Plot 2: First 3D View ---
    surf1 = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='none', alpha=0.9)
    ax1.set_xlabel('Prediction [GeV]', fontsize=label_fontsize)
    ax1.set_ylabel('Target [GeV]', fontsize=label_fontsize)
    ax1.set_zlabel('Loss', fontsize=label_fontsize)
    ax1.set_title('Loss Landscape (View 1)', fontsize=title_fontsize)
    ax1.view_init(elev=30, azim=45)
    fig1.colorbar(surf1, ax=ax1, shrink=0.6, aspect=10, pad=0.1)
    
    # --- Plot 3: Second 3D View ---
    surf2 = ax2.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='none', alpha=0.9)
    ax2.set_xlabel('Prediction [GeV]', fontsize=label_fontsize)
    ax2.set_ylabel('Target [GeV]', fontsize=label_fontsize)
    ax2.set_zlabel('Loss', fontsize=label_fontsize)
    ax2.set_title('Loss Landscape (View 2)', fontsize=title_fontsize)
    ax2.view_init(elev=60, azim=120)
    fig1.colorbar(surf2, ax=ax2, shrink=0.6, aspect=10, pad=0.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, 'loss_landscape_dual_view.png'))
    plt.close(fig1)
    
    # === SECOND FIGURE: Three 2D Histograms for Low, Mid, High Mass Ranges ===
    fig2, axs = plt.subplots(1, 3, figsize=(20, 6))
    
    cmap = plt.cm.viridis
    cmap_white_zero = cmap.copy()
    cmap_white_zero.set_under('white')
    
    r2_global = r2_score(y_true_test, y_pred_test)
    
    # --- Low mass region: 40–80 GeV ---
    hist_low = axs[0].hist2d(
        y_true_test, y_pred_test,
        bins=50,
        range=[[40, 80], [40, 80]],
        cmap=cmap_white_zero,
        norm=colors.Normalize(vmin=1)
    )
    axs[0].plot([40, 80], [40, 80], 'k--', linewidth=2)
    axs[0].set_xlabel("True Mass [GeV]", fontsize=label_fontsize)
    axs[0].set_ylabel("Predicted Mass [GeV]", fontsize=label_fontsize)
    axs[0].set_title("2D Histogram: 40–80 GeV", fontsize=title_fontsize)
    axs[0].legend(fontsize=legend_fontsize, framealpha=0.8)
    axs[0].tick_params(axis='both', labelsize=tick_fontsize)
    axs[0].grid(True, linestyle='--', alpha=0.3)
    axs[0].minorticks_on()
    fig2.colorbar(hist_low[3], ax=axs[0], label='Counts', pad=0.02, fraction=0.05).ax.tick_params(labelsize=tick_fontsize)
    
    # --- Middle mass region: 80–110 GeV ---
    hist_mid = axs[1].hist2d(
        y_true_test, y_pred_test,
        bins=100,
        range=[[80, 110], [80, 110]],
        cmap=cmap_white_zero,
        norm=colors.Normalize(vmin=1)
    )
    axs[1].plot([80, 110], [80, 110], 'k--', linewidth=2)
    axs[1].set_xlabel("True Mass [GeV]", fontsize=label_fontsize)
    axs[1].set_ylabel("Predicted Mass [GeV]", fontsize=label_fontsize)
    axs[1].set_title("2D Histogram: 80–110 GeV", fontsize=title_fontsize)
    axs[1].legend(title=f"R² = {r2_global:.3f}", fontsize=legend_fontsize, title_fontsize=legend_fontsize, framealpha=0.8)
    axs[1].tick_params(axis='both', labelsize=tick_fontsize)
    axs[1].grid(True, linestyle='--', alpha=0.3)
    axs[1].minorticks_on()
    fig2.colorbar(hist_mid[3], ax=axs[1], label='Counts', pad=0.02, fraction=0.05).ax.tick_params(labelsize=tick_fontsize)
    
    # --- High mass region: 110–1200 GeV ---
    hist_high = axs[2].hist2d(
        y_true_test, y_pred_test,
        bins=50,
        range=[[110, 1200], [110, 1200]],
        cmap=cmap_white_zero,
        norm=colors.Normalize(vmin=1)
    )
    axs[2].plot([110, 1200], [110, 1200], 'k--', linewidth=2)
    axs[2].set_xlabel("True Mass [GeV]", fontsize=label_fontsize)
    axs[2].set_ylabel("Predicted Mass [GeV]", fontsize=label_fontsize)
    axs[2].set_title("2D Histogram: 110–1200 GeV", fontsize=title_fontsize)
    axs[2].legend(fontsize=legend_fontsize, framealpha=0.8)
    axs[2].tick_params(axis='both', labelsize=tick_fontsize)
    axs[2].grid(True, linestyle='--', alpha=0.3)
    axs[2].minorticks_on()
    fig2.colorbar(hist_high[3], ax=axs[2], label='Counts', pad=0.02, fraction=0.05).ax.tick_params(labelsize=tick_fontsize)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, '2d_hist_by_mass_range.png'))
    plt.close(fig2)

    # === THIRD FIGURE: Three Mass Range Histograms ===
    fig3, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Coarse bins: 40–80
    axs[0].hist(y_true_test, bins=20, range=[40, 80], histtype='step', color='orange', label='Truth', linewidth=linewidth)
    axs[0].hist(y_pred_test, bins=20, range=[40, 80], histtype='step', color='blue', label='Prediction', linewidth=linewidth)
    axs[0].set_xlabel('Mass [GeV]', fontsize=label_fontsize)
    axs[0].set_ylabel('Counts', fontsize=label_fontsize)
    axs[0].set_title('Mass Range: 40–80 GeV', fontsize=title_fontsize)
    axs[0].legend(fontsize=legend_fontsize, framealpha=0.8)
    axs[0].grid(True, linestyle='--', alpha=0.3)
    axs[0].tick_params(axis='both', labelsize=tick_fontsize)
    axs[0].minorticks_on()

    # Fine bins: 80–110
    axs[1].hist(y_true_test, bins=100, range=[80, 110], histtype='step', color='orange', label='Truth', linewidth=linewidth)
    axs[1].hist(y_pred_test, bins=100, range=[80, 110], histtype='step', color='blue', label=f'Prediction, MedianAE: {median_ae:.3f}', linewidth=linewidth)
    axs[1].set_xlabel('Mass [GeV]', fontsize=label_fontsize)
    axs[1].set_ylabel('Counts', fontsize=label_fontsize)
    axs[1].set_title('Mass Range: 80–110 GeV', fontsize=title_fontsize)
    axs[1].legend(fontsize=legend_fontsize, framealpha=0.8)
    axs[1].grid(True, linestyle='--', alpha=0.3)
    axs[1].tick_params(axis='both', labelsize=tick_fontsize)
    axs[1].minorticks_on()

    # Coarse bins: 110–1200
    axs[2].hist(y_true_test, bins=30, range=[110, 1200], histtype='step', color='orange', label='Truth', linewidth=linewidth)
    axs[2].hist(y_pred_test, bins=30, range=[110, 1200], histtype='step', color='blue', label='Prediction', linewidth=linewidth)
    axs[2].set_xlabel('Mass [GeV]', fontsize=label_fontsize)
    axs[2].set_ylabel('Counts', fontsize=label_fontsize)
    axs[2].set_title('Mass Range: 110–1200 GeV', fontsize=title_fontsize)
    axs[2].legend(fontsize=legend_fontsize, framealpha=0.8)
    axs[2].grid(True, linestyle='--', alpha=0.3)
    axs[2].tick_params(axis='both', labelsize=tick_fontsize)
    axs[2].minorticks_on()

    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, 'mass_histograms_ranges.png'))
    plt.close(fig3)


def plot_distributions(signal_data, x_range=None, variable_name='', n_bins=40, title='',save_path=None, figsize=(10, 6)):
    plt.figure(figsize=figsize, dpi=100)

    signal_data = np.array(signal_data)

    # Extract 1D data for a specific feature & particle
    if variable_name != '':
        feature_index = input_keys_dict[variable_name[:-1]]
        particle_index = int(variable_name[-1])
        signal_data = signal_data[:, particle_index, feature_index]

    # Handle NaNs and Infs
    n_nan = np.isnan(signal_data).sum()
    n_inf = np.isinf(signal_data).sum()
    signal_data_clean = signal_data[~np.isnan(signal_data) & ~np.isinf(signal_data)]

    if x_range is None:
        x_range_min = np.nanmin(signal_data_clean)
        x_range_max = np.nanmax(signal_data_clean)
        x_range = [x_range_min, x_range_max]

    # Basic stats
    min_val = np.min(signal_data_clean)
    max_val = np.max(signal_data_clean)
    mean_val = np.mean(signal_data_clean)

    underflow = np.sum(signal_data_clean < x_range[0])
    overflow = np.sum(signal_data_clean > x_range[1])

    # Plot histogram
    plt.hist(signal_data_clean, bins=n_bins, range=x_range, histtype='step',
             label=(
                 f"Signal\n"
                 f"Total: {len(signal_data_clean)}\n"
                 f"NaN: {n_nan}, Inf: {n_inf}\n"
                 f"Underflow: {underflow}, Overflow: {overflow}\n"
                 f"Min: {min_val:.3f}, Max: {max_val:.3f}, Mean: {mean_val:.3f}"
             ),
             density=True)

    plt.xlabel(variable_name)
    plt.ylabel('Density')

    if title:
        plt.title(title)
    else:
        plt.title(f'Distribution of {variable_name}')

    if (x_range[0] != x_range[1]):
        plt.xlim(*x_range)

    plt.legend(loc='best')
    plt.grid(True)

    # === Save if path is provided ===
    if save_path:
        plt.savefig(save_path, dpi=100)
        plt.close()
    else: 
        plt.show()

def plot_all_input_data(input_data, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    
    for index, col in enumerate(input_data.T):  # Have to transpose to plot singular features
        plot_distributions(col, save_path=f'{save_folder}/col_{index}')

    print("Finished Plotting")

def plot_distributions_by_range(all_y, y_train, ranges, bins=100, save_folder='Plots'):
    all_y = np.array(all_y)
    y_train = np.array(y_train)

    # Create folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    n_ranges = len(ranges)
    fig, axs = plt.subplots(1, n_ranges, figsize=(6 * n_ranges, 5), constrained_layout=True)

    if n_ranges == 1:
        axs = [axs]  # Ensure axs is iterable

    for i, (xmin, xmax) in enumerate(ranges):
        ax = axs[i]

        ax.hist(all_y, bins=bins, range=(xmin, xmax), alpha=0.6, color='orange', label='Sampled batch y', edgecolor='black')
        ax.hist(y_train, bins=bins, range=(xmin, xmax), alpha=0.4, color='blue', label='Original y_train', edgecolor='black')

        ax.set_title(f'{xmin}–{xmax} GeV')
        ax.set_xlabel('Invariant Mass [GeV]')
        ax.set_ylabel('Frequency')
        ax.set_xlim(xmin, xmax)
        ax.grid(True)
        ax.legend()

    # Save the combined figure
    output_path = os.path.join(save_folder, "distribution_comparison_all_ranges.png")
    plt.savefig(output_path)
    print(f"Saved combined plot to: {output_path}")
    plt.close(fig)

def plot_particle_distributions(peak_indices, signal_data, x_range=None, title='', n_bins=40, folder_name=''):
    signal_data = np.array(signal_data)

    one_dimensional_data = signal_data.ndim != 2

    if one_dimensional_data:
        n_particles = 1
    else:
        n_particles = signal_data.shape[1]

    # Naming convention
    base_names = ['lep', 'tau', 'MET']

    if one_dimensional_data:
        particle_names = ['']
    else:
        particle_names = base_names + [f'jet{i+1}' for i in range(n_particles - len(base_names))]

    # Create one figure with subplots side-by-side
    fig, axs = plt.subplots(1, n_particles, figsize=(5 * n_particles, 5), dpi=100)

    if n_particles == 1:
        axs = [axs]  # Ensure iterable
        signal_data = np.array([signal_data]).reshape(-1, 1)

    for i in range(n_particles):
        ax = axs[i]

        particle_data = signal_data[:, i]

        # Count NaNs and Infs
        num_nans = np.sum(np.isnan(particle_data))
        num_infs = np.sum(np.isinf(particle_data))

        # Handle NaNs and Infs
        valid_mask = ~np.isnan(particle_data) & ~np.isinf(particle_data)
        particle_data_clean = particle_data[valid_mask]
        valid_peak_indices = peak_indices[valid_mask]

        # Min and Max
        data_min = np.min(particle_data_clean)
        data_max = np.max(particle_data_clean)

        # Determine range if not given
        if x_range is None:
            current_x_range = [data_min, data_max]
        else:
            current_x_range = x_range

        # Count underflow and overflow
        underflow = np.sum(particle_data_clean < current_x_range[0])
        overflow = np.sum(particle_data_clean > current_x_range[1])

        # Categorize data
        peak_data = particle_data_clean[valid_peak_indices]
        tail_data = particle_data_clean[~valid_peak_indices]

        # Plot each particle
        ax.hist(peak_data, bins=n_bins, range=current_x_range, histtype='step', density=True,
                color='blue', linestyle='solid',
                label=(f'{particle_names[i]} Peak [88, 95]\n'
                       f'NaNs: {num_nans}, Infs: {num_infs}\n'
                       f'Under: {underflow}, Over: {overflow}\n'
                       f'Min: {data_min:.2f}, Max: {data_max:.2f}'))

        ax.hist(tail_data, bins=n_bins, range=current_x_range, histtype='step', density=True,
                color='red', linestyle='dashed',
                label=f'{particle_names[i]} Tail (<88 or >95)')

        ax.set_xlabel('Value')
        ax.set_title(f'Distribution of {particle_names[i]}')
        ax.grid(True)
        ax.legend(loc='best')

    axs[0].set_ylabel('Density')
    plt.tight_layout()

    # Save figure
    os.makedirs(folder_name, exist_ok=True)
    filename = title.replace(" ", "_").lower() if title else "particle_distribution"
    save_path = os.path.join(folder_name, f"{filename}.png")
    plt.savefig(save_path)
    plt.close()


# ================================= #
# ============= DNN =============== #
# ================================= #

class SimpleDNN(nn.Module):
    def __init__(self, n_inputs, hidden_layers, dropout_prob=0.1):
        super(SimpleDNN, self).__init__()

        # Complete layer sizes: [n_inputs, hidden1, hidden2, ..., 1]
        layer_sizes = [n_inputs] + hidden_layers + [1]

        # Create a list of Linear layers
        self.layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:])
        ])

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.dropout(F.relu(layer(x)))
        y = F.softplus(self.layers[-1](x))
        return y

def permutation_feature_importance(model, val_loader, baseline_loss, device, criterion):
    model.eval()
    X_val, y_val = [], []
    for inputs, targets in val_loader:
        X_val.append(inputs.cpu().numpy())
        y_val.append(targets.cpu().numpy())
    X_val = np.concatenate(X_val)
    y_val = np.concatenate(y_val)

    importances = []
    for col in range(X_val.shape[1]):
        X_val_permuted = X_val.copy()
        np.random.shuffle(X_val_permuted[:, col])
        total_loss = 0.0

        with torch.no_grad():
            for i in range(0, len(X_val), val_loader.batch_size):
                inputs = torch.tensor(X_val_permuted[i:i+val_loader.batch_size], dtype=torch.float32).to(device)
                targets = torch.tensor(y_val[i:i+val_loader.batch_size], dtype=torch.float32).to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                total_loss += loss.item()

        avg_loss = total_loss / (len(X_val) // val_loader.batch_size)
        importances.append(avg_loss - baseline_loss)

    return np.array(importances)

class WeightedTailLoss(nn.Module):
    def __init__(self, base_loss=None, mass_weight=[2.0, 2.0], threshold=[85.0, 100.0]):
        super().__init__()
        self.base_loss = base_loss if base_loss else nn.SmoothL1Loss(reduction='none')
        self.mass_weight = mass_weight  # [left_tail_weight, right_tail_weight]
        self.threshold = threshold      # [left_threshold, right_threshold]

        if hasattr(self.base_loss, 'reduction'):
            self.base_loss.reduction = 'none'

    def forward(self, input, target):
        loss = self.base_loss(input, target)

        # Masks for left and right tail regions
        left_mask  = target < self.threshold[0]
        right_mask = target > self.threshold[1]

        # Apply separate weights
        weighted_loss = torch.where(left_mask,  loss * self.mass_weight[0], loss)
        weighted_loss = torch.where(right_mask, weighted_loss * self.mass_weight[1], weighted_loss)

        return weighted_loss.mean()



class QuantileAwareLoss(nn.Module):
    def __init__(self, quantile_edges, base_loss=nn.L1Loss(), alpha=0.5, squared=False):
        super().__init__()
        self.register_buffer('quantile_edges', torch.tensor(quantile_edges, dtype=torch.float32))
        self.base_loss = base_loss
        self.alpha = alpha
        self.squared = squared

    def forward(self, preds, targets):
        preds = preds.squeeze()
        targets = targets.squeeze()

        # Ensure quantile_edges are on the same device
        quantile_edges = self.quantile_edges.to(targets.device)
        num_bins = len(quantile_edges) - 1

        # Compute base loss (OldLoss)
        base_loss_value = self.base_loss(preds, targets)

        # Compute quantile bin penalty (QuantileLoss)
        target_bins = torch.bucketize(targets, quantile_edges)
        pred_bins = torch.bucketize(preds, quantile_edges)

        # Normalize by the number of bins
        if self.squared:
            quantile_loss_value = ((pred_bins - target_bins) ** 2 / num_bins**2).float().mean()
        else:
            quantile_loss_value = torch.abs( (pred_bins - target_bins) / num_bins ).float().mean()
            

        # Combine losses
        total_loss = self.alpha * base_loss_value + (1 - self.alpha) * quantile_loss_value
        return total_loss


class InverseGaussianWeightedLoss(nn.Module):
    def __init__(self, base_loss=nn.SmoothL1Loss(beta=10, reduction='mean'), center=91.0, sigma=5.0, max_weight=10.0):
        super().__init__()
        self.base_loss = base_loss
        self.center = center
        self.sigma = sigma
        self.max_weight = max_weight

        if hasattr(self.base_loss, 'reduction'):
            self.base_loss.reduction = 'none' 

    def forward(self, input, target):
        input = input.squeeze()
        target = target.squeeze()

        loss = self.base_loss(input, target)

        gaussian = torch.exp(-0.5 * ((target - self.center) / self.sigma) ** 2)
        weights = 1.0 / (gaussian + 1e-8)  # inverse Gaussian

        # Cap weights at max_weight
        weights = torch.clamp(weights, max=self.max_weight)

        weighted_loss = loss * weights
        return weighted_loss.mean()


# ================================ #
# ============ MISC ============== #
# ================================ #
import re

def replace_particle_indices(feature_names, particle_labels, avoid_features):
    new_feature_names = []
    num_labels = len(particle_labels)

    for index, name in enumerate(feature_names):
        match = re.search(r'_x(\d+)$', name)
        if match:
            index = int(match.group(1))
            particle_name = particle_labels[index % num_labels]
            # Remove __x<number> entirely, then append _<particle_name>
            base_name = re.sub(r'_x\d+$', '', name)

            if index in avoid_features: new_name = f'{base_name}' # Dont add the particle name in the avoided features
            else: new_name = f'{base_name}_{particle_name}'
                
            new_feature_names.append(new_name)
        else:
            new_feature_names.append(name)  # No match, keep as is

    return new_feature_names


    