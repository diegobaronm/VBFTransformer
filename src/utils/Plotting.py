# This is the main plotting module for the Regression DNN and Transformer models, plotting functions are called when training
# the model (input variables are plotted) and when testing the performance of the model (metrics and predictions are plotted)

# Notably this code is quite long and complex because of the several different plots that each model demands and could be restructured futher 
# to improve readability. For now helper functions are defined in PlottingHelpers.py and functions which name starts with _ are private and only meant to 
# be used within this script.

# === Third-Party Libraries ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# === PyTorch and Related ===
import torch
from loguru import logger
from pathlib import Path

import src.utils.PlottingHelpers as plotHelp

DEFAULT_STYLE = {
    'title_fontsize': 16,
    'label_fontsize': 14,
    'legend_fontsize': 12,
    'tick_fontsize': 12,
    'linewidth': 2
}

def plot_particle_distributions(left_tail_ind, right_tail_ind, signal_data, x_range=None, title='', titles=None, n_bins=40, no_particle_name=False, folder_name=''):
    """Plots input training data and separates it into left, peak and right tail to visually discriminate variables"""
    if signal_data.ndim == 1:
        # single feature (1D input)
        signal_data = signal_data[:, None]   # shape (n_events, 1)
        n_particles = 1
    else:
        # already 2D
        signal_data = np.atleast_2d(signal_data)
        n_particles = signal_data.shape[1]

    # If we have many particles, then their names will be lep, tau, met, jet1, jet2 etc. 
    base_names = ['lep', 'tau', 'MET'] + [f'jet{i+1}' for i in range(n_particles - 3)]

    # we also need to ensure not to include tau or met if only 1 or 2 particles used
    particle_names = [''] if no_particle_name else base_names[:n_particles]

    fig, axs = plt.subplots(1, n_particles, figsize=(5 * n_particles, 5), dpi=100)
    axs = np.atleast_1d(axs)

    for i, ax in enumerate(axs):
        data = signal_data[:, i]
        valid = ~np.isnan(data) & ~np.isinf(data)
        clean = data[valid]
        
        num_nans, num_infs = np.isnan(data).sum(), np.isinf(data).sum()
        l_mask, r_mask = left_tail_ind[valid], right_tail_ind[valid]
        p_mask = ~(l_mask | r_mask)

        dmin, dmax = clean.min(), clean.max()
        rng = [dmin, dmax] if x_range is None else x_range
        under, over = (clean < rng[0]).sum(), (clean > rng[1]).sum()

        # Plot
        ax.hist(clean[l_mask], bins=n_bins, range=rng, histtype='step', density=True, color='purple', linestyle='dashdot', linewidth=2.0,
                label=f'{particle_names[i]} Left Tail')
        ax.hist(clean[p_mask],  bins=n_bins, range=rng, histtype='step', density=True, color='blue', linestyle='solid',  linewidth=2,
                label=(f'{particle_names[i]} Peak\nNaNs: {num_nans}, Infs: {num_infs}\n' f'Under: {under}, Over: {over}\nMin: {dmin:.2f}, Max: {dmax:.2f}'))
        ax.hist(clean[r_mask],  bins=n_bins, range=rng, histtype='step', density=True,
                color='red', linestyle='dashed', linewidth=2, label=f'{particle_names[i]} Right Tail')

        ax.set_xlabel('Value')
        ax.set_title(titles[i] if titles is not None else title if no_particle_name else f'{particle_names[i]} {title}')
        ax.grid(True); ax.legend(loc='best')

    axs[0].set_ylabel('Density')
    plt.tight_layout()

    filename = title.replace(" ", "_").lower() if title else "particle_distribution"
    plotHelp.save_fig(fig, Path(folder_name) / filename, dpi=100)
    plt.close()

def plot_metrics(train_losses, val_losses, y_true_test, y_pred_test, criterion, folder_name="plots"):
    """Plot training metrics with loss curves, 2D histograms, and mass distributions."""
    
    # Compute derived metrics
    residuals = y_pred_test - y_true_test
    median_abs_error = np.median(np.abs(residuals))
    r2_global = r2_score(y_true_test, y_pred_test)
    
    # Create all plots
    _plot_loss_and_landscape(train_losses, val_losses, criterion, folder_name, DEFAULT_STYLE)
    _plot_2d_histograms(y_true_test, y_pred_test, r2_global, folder_name, DEFAULT_STYLE)
    _plot_mass_distributions(y_true_test, y_pred_test, median_abs_error, folder_name, DEFAULT_STYLE)


def _plot_loss_and_landscape(train_losses, val_losses, criterion, folder_name, style):
    """Create loss curves and 3D loss landscape plots to view how loss function behaves"""
    
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3)
    
    # Loss curves
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.plot(np.log(train_losses), label='Train Loss', color='red', linewidth=style['linewidth'])
    ax_loss.plot(np.log(val_losses), label='Validation Loss', color='green', linewidth=style['linewidth'])
    plotHelp.style_axis(ax_loss, 'Epoch', 'Log(Loss)', 'Train & Val Loss (Log Scale)', style)
    
    # Compute loss landscape
    loss_grid, X, Y = plotHelp.compute_loss_landscape(criterion)
    
    # 3D landscapes with different views
    views = [(30, 45, 'View 1'), (60, 120, 'View 2')]
    for i, (elev, azim, view_name) in enumerate(views):
        ax_3d = fig.add_subplot(gs[0, i + 1], projection='3d')
        surf = ax_3d.plot_surface(X, Y, loss_grid, cmap=cm.viridis, edgecolor='none', alpha=0.9)
        ax_3d.set_xlabel('Prediction [GeV]', fontsize=style['label_fontsize'])
        ax_3d.set_ylabel('Target [GeV]', fontsize=style['label_fontsize'])
        ax_3d.set_zlabel('Loss', fontsize=style['label_fontsize'])
        ax_3d.set_title(f'Loss Landscape ({view_name})', fontsize=style['title_fontsize'])
        ax_3d.view_init(elev=elev, azim=azim)
        fig.colorbar(surf, ax=ax_3d, shrink=0.6, aspect=10, pad=0.1)
    
    plt.subplots_adjust(wspace=0.4)
    plotHelp.save_fig(fig, Path(folder_name) / 'loss_landscape_dual_view.png', dpi=100)
    plt.close(fig)

def _calculate_min_max_data(y_one, y_two, rounding=False):
    """Calculates the integer min and max between the train and test data so plots can be adjusted"""
    minimum = np.floor(min(np.min(y_one), np.min(y_two)))
    maximum = np.ceil(max(np.max(y_one), np.max(y_two)))

    if rounding: 
        return int(minimum), int(maximum)
        
    return minimum, maximum

def _plot_2d_histograms(y_true_test, y_pred_test, r2_global, folder_name, style):
    """Create 2D histograms for different mass ranges."""

    min_y, max_y = _calculate_min_max_data(y_true_test, y_pred_test, rounding=True)
    fig, axs = plt.subplots(1, 4, figsize=(27, 6))
    
    # Define histogram configurations
    hist_configs = [
        {'range': [[min_y, 80], [min_y, 80]], 'bins': 50, 'title': f'2D Histogram: {min_y}–80 GeV'},
        {'range': [[80, 110], [80, 110]], 'bins': 100, 'title': '2D Histogram: 80–110 GeV'},
        {'range': [[110, max_y], [110, max_y]], 'bins': 50, 'title': f'2D Histogram: 110–{max_y} GeV'}
    ]
    
    cmap = plotHelp.get_colormap_with_white_zero()
    
    # Regular histograms
    for i, config in enumerate(hist_configs):
        _create_2d_histogram(axs[i], y_true_test, y_pred_test, config, cmap, fig, style)
        if i == 1:  # Add R² to middle plot
            axs[i].plot([], [], ' ', label=f"R² = {r2_global:.3f}")
            axs[i].legend(fontsize=style['legend_fontsize'], loc='lower right', frameon=True)
    
    # Log-scale histogram
    _create_log_histogram(axs[3], y_true_test, y_pred_test, cmap, fig, style)
    
    plt.tight_layout()
    plotHelp.save_fig(fig, Path(folder_name) / '2d_hist_by_mass_range.png', dpi=100)
    plt.close(fig)


def _plot_mass_distributions(y_true_test, y_pred_test, median_ae, folder_name, style):
    """Create mass distribution histograms for different ranges."""
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    min_y, max_y = _calculate_min_max_data(y_true_test, y_pred_test, rounding=True)

    # Define histogram configurations
    hist_configs = [
        {'bins': 40, 'range': [40, 80], 'title': f'{min_y}–80 GeV', 'show_median': False},
        {'bins': 120, 'range': [80, 110], 'title': '80–110 GeV', 'show_median': True},
        {'bins': 40, 'range': [110, 1200], 'title': f'110–{max_y} GeV', 'show_median': False}
    ]
    
    for i, config in enumerate(hist_configs):
        _create_mass_histogram(axs[i], y_true_test, y_pred_test, config, median_ae, style)
    
    plt.tight_layout()
    plotHelp.save_fig(fig, Path(folder_name) / 'mass_histograms_ranges.png', dpi=100)
    plt.close(fig)

def _create_2d_histogram(ax, y_true, y_pred, config, cmap, fig, style):
    """Create a single 2D histogram with consistent styling."""
    
    hist = ax.hist2d(y_true, y_pred,bins=config['bins'],range=config['range'],cmap=cmap,norm=colors.Normalize(vmin=1))
    
    # Add diagonal reference line for correlation coefficient
    range_min, range_max = config['range'][0][0], config['range'][0][1]
    ax.plot([range_min, range_max], [range_min, range_max], 'k--', linewidth=2)
    
    # Style the plot
    ax.set_xlabel("True Mass [GeV]", fontsize=style['label_fontsize'])
    ax.set_ylabel("Predicted Mass [GeV]", fontsize=style['label_fontsize'])
    ax.set_title(config['title'], fontsize=style['title_fontsize'])
    ax.tick_params(axis='both', labelsize=style['tick_fontsize'])
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.minorticks_on()
    
    fig.colorbar(hist[3], ax=ax, label='Counts', pad=0.02, fraction=0.05).ax.tick_params(labelsize=style['tick_fontsize'])


def _create_log_histogram(ax, y_true, y_pred, cmap, fig, style):
    """Create log-scale 2D histogram."""
    
    # Compute log-scale bounds and allow for extra room for neater visuals
    low, high = _calculate_min_max_data(y_pred, y_true, rounding=False)
    eps = 1e-9

    # Safeguarding when training low epochs causing model to predict -inf for some masses.
    if low <= 0:
        low = eps
    low = np.log(0.95 * low)

    high = np.log(1.05 * high)
    
    hist = ax.hist2d(np.log(y_true), np.log(y_pred), bins=100, range=[[low, high], [low, high]], cmap=cmap, norm=colors.LogNorm(vmin=1))
    
    # Add diagonal reference line for correlation coefficient
    ax.plot([low, high], [low, high], 'k--', linewidth=2)
    
    # Style the plot
    ax.set_xlabel("Ln True Mass [GeV]", fontsize=style['label_fontsize'])
    ax.set_ylabel("Ln Predicted Mass [GeV]", fontsize=style['label_fontsize'])
    ax.set_title("Ln 2D Histogram [GeV]", fontsize=style['title_fontsize'])
    ax.tick_params(axis='both', labelsize=style['tick_fontsize'])
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.minorticks_on()
    
    fig.colorbar(hist[3], ax=ax, label='Log Counts', pad=0.02, fraction=0.05).ax.tick_params(labelsize=style['tick_fontsize'])


def _create_mass_histogram(ax, y_true, y_pred, config, median_ae, style):
    """Create a single mass distribution histogram."""
    
    bin_width = (config['range'][1] - config['range'][0]) / config['bins']
    
    ax.hist(y_true, bins=config['bins'], range=config['range'], 
            histtype='step', color='orange', label='Truth', linewidth=style['linewidth'])
    
    # Add median AE to legend if specified
    pred_label = f'Prediction, MedianAE: {median_ae:.3f}' if config['show_median'] else 'Prediction'
    ax.hist(y_pred, bins=config['bins'], range=config['range'], 
            histtype='step', color='blue', label=pred_label, linewidth=style['linewidth'])
    
    # Style the plot
    ax.set_xlabel('Mass [GeV]', fontsize=style['label_fontsize'])
    ax.set_ylabel('Counts', fontsize=style['label_fontsize'])
    ax.set_title(f'Mass Range: {config["title"]} (Bin width: {bin_width:.1f} GeV)', 
                 fontsize=style['title_fontsize'])
    ax.legend(fontsize=style['legend_fontsize'], framealpha=0.8)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(axis='both', labelsize=style['tick_fontsize'])
    ax.minorticks_on()

def plot_and_save_attention(attn_per_example, save_dir):
    """Visualizes how the attention flows between tokens (rollout), and their accumulated attention displaying the general importance
       of the token"""
    
    # --- extract per-layer numpy attention matrices ---
    layer_mats = []
    for attn in attn_per_example:
        arr = attn.detach().cpu().numpy()
        if arr.ndim > 2:
            arr = arr.mean(axis=tuple(range(arr.ndim - 2)))
        layer_mats.append(arr)

    # --- compute rollout ---
    seq_len = layer_mats[0].shape[0]
    aug = [(m + np.eye(seq_len)) / (m.sum(axis=-1, keepdims=True) + 1e-6) for m in layer_mats]
    rollout = aug[0]
    for m in aug[1:]:
        rollout = m @ rollout

    # --- labels ---
    num_jets = seq_len - 3
    labels = ["lep", "tau", "MET"] + [f"jet{i+1}" for i in range(num_jets)]
    labels = labels[:seq_len] # Clip the array if less than 3 particles used

    # --- rollout heatmap ---
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    im = ax1.imshow(rollout, aspect='auto')
    ax1.set_xticks(range(seq_len)); ax1.set_yticks(range(seq_len))
    ax1.set_xticklabels(labels, rotation=45, ha="right"); ax1.set_yticklabels(labels)
    ax1.set(title='Attention Rollout Heatmap', xlabel='Source Token', ylabel='Target Token')
    fig1.colorbar(im, ax=ax1, label='Rollout Weight')
    fig1.tight_layout()
    plotHelp.save_fig(fig1, Path(save_dir) / 'rollout.png', dpi=100)
    plt.close(fig1)

    # --- token importance ---
    importance = rollout.sum(axis=0)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(range(seq_len), importance)
    ax2.set_xticks(range(seq_len)); ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set(title='Token Importance from Rollout', xlabel='Token', ylabel='Accumulated Importance')
    fig2.tight_layout()
    plotHelp.save_fig(fig2, Path(save_dir) / 'importance.png', dpi=100)
    plt.close(fig2)


def permutation_feature_importance_fast(model, val_loader, metric_fn, device, higher_is_better=True, batch_size=1024):
    """Computes the importance of inputs in a DNN by checking how the metric changes when the inputs are shuffled
       For simplicity this function does shuffle blindly so unphysical input data can arise (eg. events with total charge)
       It is up to the user to understand that when these cases arise."""
    
    # Preload entire dataset once onto GPU as a single tensor
    X_list, y_list = [], []
    for Xb, yb in val_loader:
        X_list.append(Xb)
        y_list.append(yb)
    X_val_t = torch.cat(X_list, dim=0).to(device)         # shape (num_events,...) on GPU
    y_true  = torch.cat(y_list, dim=0).to(device).squeeze()

    num_events = X_val_t.shape[0]

    _, num_features = X_val_t.shape
    importances = torch.zeros(num_features, device=device)
        
    # Compute baseline prediction once
    with torch.no_grad():
        preds_base = model(X_val_t).squeeze()
    base_score = metric_fn(y_true.cpu().numpy(), preds_base.cpu().numpy())

    # Helper to run batched prediction on a GPU tensor
    def batched_predict(X):
        out = []
        with torch.no_grad():
            for i in range(0, num_events, batch_size):
                out.append(model(X[i : i + batch_size]))
        return torch.cat(out).squeeze()

    # In‐place shuffle + restore for each feature (or token×feature)
    for f in range(num_features):
        backup = X_val_t[:, f].clone()
        perm   = torch.randperm(num_events, device=device)
        X_val_t[:, f] = X_val_t[perm, f]

        preds_p = batched_predict(X_val_t)
        score_p = metric_fn(y_true.cpu().numpy(), preds_p.cpu().numpy())
        delta   = (base_score - score_p) if higher_is_better else (score_p - base_score)
        importances[f] = delta

        X_val_t[:, f] = backup

    return importances.cpu().numpy()


def compute_feature_importance_and_correlation_plot(model, datamodule, device, result_dir,feature_names, trainer=None):
    """Computes importances for MSE, R2 and MeanAbsError for DNN input data"""
    
    val_loader = trainer.datamodule.val_dataloader() if trainer else datamodule.val_dataloader()

    # Load validation data
    X_val, y_val = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            X_val.append(inputs.cpu().numpy())
            y_val.append(targets.cpu().numpy())
    X_val = np.concatenate(X_val)
    y_val = np.concatenate(y_val).squeeze()

    # Baseline predictions and metrics
    inputs_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(inputs_t).cpu().numpy().squeeze()
    
    # Compute permutation importances for each metric
    imp_mse = permutation_feature_importance_fast(model, val_loader, mean_squared_error, device, higher_is_better=False)
    imp_r2 = permutation_feature_importance_fast(model, val_loader, r2_score, device, higher_is_better=True)
    imp_mae = permutation_feature_importance_fast(model, val_loader, mean_absolute_error, device, higher_is_better=False)

    # Simple bar chart
    plot_labels_and_data = zip(["mse", "mae", "r2"], [imp_mse, imp_mae, imp_r2],
        ["Increase in MSE after permutation", "Increase in MAE after permutation", "Decrease in R² after permutation"]
                              )
    for metric_name, importance_vector, xlabel in plot_labels_and_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(feature_names, importance_vector, color='steelblue')
        ax.set_xlabel(xlabel)
        ax.set_title(f"Feature Importances ({metric_name.upper()})")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{result_dir}/feature_importance_{metric_name}.png", dpi=150)
        plt.close(fig)


def plot_resampled_distributions(y_train, y_train_inverse_sampled, save_path):
    """Plots KDE and original batches to view how the 'new' input data looks"""
    
    # Define zoomed-in regions
    low, high = _calculate_min_max_data(y_pred, y_true, rounding=True)
    regions = [(low, 80), (80, 110), (110, high)]
    region_titles = [f"Low Range ({low}–80) GeV", "Mid Range (80–110) GeV", f"High Range (110–{high}) GeV"]

    # Set up compact layout
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    # Because train loaders drop events which fall outside the last batch, this cutting needs to be done
    # to ensure both datasets are the same size as we are comparing counts
    min_data_points = min(len(y_train), len(y_train_inverse_sampled))
    y_train = y_train[:min_data_points]
    y_train_inverse_sampled = y_train_inverse_sampled[:min_data_points]

    for ax, (low, high), title in zip(axes, regions, region_titles):
        ax.hist(y_train, bins=50, range=(low, high), alpha=0.6, label='Original', color='steelblue', density=True)
        ax.hist(y_train_inverse_sampled, bins=50, range=(low, high), alpha=0.6, label='Resampled', color='darkorange', density=True)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Target Value')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    axes[0].set_ylabel('Density')
    axes[1].legend(loc='upper right', fontsize=10)
    
    plotHelp.save_fig(fig, Path(save_path), dpi=100)


def plot_kde_and_inverse_weights(y_train, density, inv_density, save_path=None, xlim=(60, 120)):
    """Plots how the data and KDE weights look, this shows the inverse sampling distribution"""
    
    sorted_idx = np.argsort(y_train)
    y_sorted = y_train[sorted_idx]
    density_sorted = density[sorted_idx]
    inv_density_sorted = inv_density[sorted_idx]

    # Set up figure
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Histogram of y_train
    ax1.hist(y_train, bins=1000, density=True, alpha=0.4, label='Histogram of y_train')
    ax1.plot(y_sorted, density_sorted, color='blue', label='KDE (Density Estimate)', linewidth=2)

    # Inverse density on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(y_sorted, inv_density_sorted, color='red', label='Inverse Density (Sampling Weight)', linewidth=2, linestyle='--')

    # Labels
    ax1.set_xlabel('Target Value (MZ_TRUTH)', fontsize=12)
    ax1.set_ylabel('Density', color='blue', fontsize=12)
    ax2.set_ylabel('Inverse Density Weight', color='red', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')

    # Titles and legends
    plt.title("KDE of y_train and Inverse Sampling Weights", fontsize=14)
    fig.legend(loc='upper right', bbox_to_anchor=(0.88, 0.88))
    plt.xlim(xlim)
    plt.grid(True)
    
    plotHelp.save_fig(fig, Path(save_path), dpi=100)
    plt.close()
