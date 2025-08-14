# === Standard Library ===
import os

# === Third-Party Libraries ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# === PyTorch and Related ===
import torch
from torch.utils.data import DataLoader
from loguru import logger


def plot_particle_distributions(left_tail_indices, right_tail_indices, signal_data, x_range=None, title='', titles=None, n_bins=40, no_particle_name=False, folder_name=''):
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

        valid_left_indices = left_tail_indices[valid_mask]
        valid_right_indices = right_tail_indices[valid_mask]
        valid_peak_indices = ~(valid_left_indices | valid_right_indices)

        # Min and Max
        data_min = np.min(particle_data_clean)
        data_max = np.max(particle_data_clean)

        # Determine range if not given
        current_x_range = [data_min, data_max] if x_range is None else x_range

        # Count underflow and overflow
        underflow = np.sum(particle_data_clean < current_x_range[0])
        overflow = np.sum(particle_data_clean > current_x_range[1])

        # Categorize data
        left_data = particle_data_clean[valid_left_indices]
        peak_data = particle_data_clean[valid_peak_indices]
        right_data = particle_data_clean[valid_right_indices]

        # Plot each category
        ax.hist(left_data, bins=n_bins, range=current_x_range, histtype='step', density=True,
                color='darkgreen', linestyle='dotted', linewidth=2, label=f'{particle_names[i]} Left Tail')

        ax.hist(peak_data, bins=n_bins, range=current_x_range, histtype='step', density=True,
                color='blue', linestyle='solid', linewidth=2,
                label=(f'{particle_names[i]} Peak\n'
                       f'NaNs: {num_nans}, Infs: {num_infs}\n'
                       f'Under: {underflow}, Over: {overflow}\n'
                       f'Min: {data_min:.2f}, Max: {data_max:.2f}'))

        ax.hist(right_data, bins=n_bins, range=current_x_range, histtype='step', density=True,
                color='red', linestyle='dashed', linewidth=2, label=f'{particle_names[i]} Right Tail')

        ax.set_xlabel('Value')

        if titles is not None: 
            ax.set_title(f'{titles[i]}')
        elif no_particle_name: 
            ax.set_title(f'{title}')
        else:
            ax.set_title(f'{particle_names[i]} {title}')
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
    
    plt.subplots_adjust(wspace=0.4)
    plt.savefig(os.path.join(folder_name, 'loss_landscape_dual_view.png'))
    plt.close(fig1)
    
    # === SECOND FIGURE: Four 2D Histograms for Low, Mid, High Mass Ranges ===
    fig2, axs = plt.subplots(1, 4, figsize=(27, 6))
    
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
    
    # 👉 Add R² to the legend
    axs[1].plot([], [], ' ', label=f"R² = {r2_global:.3f}")
    axs[1].legend(fontsize=legend_fontsize, loc='lower right', frameon=True)
    
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
    axs[2].tick_params(axis='both', labelsize=tick_fontsize)
    axs[2].grid(True, linestyle='--', alpha=0.3)
    axs[2].minorticks_on()
    fig2.colorbar(hist_high[3], ax=axs[2], label='Counts', pad=0.02, fraction=0.05).ax.tick_params(labelsize=tick_fontsize)


    # Determine log-scale axis bounds with padding
    low = np.log(min(np.min(y_pred_test), np.min(y_true_test)) * 0.95)
    high = np.log(max(np.max(y_pred_test), np.max(y_true_test)) * 1.05)
    
    # 2D histogram in log-space with log color scale
    hist_general = axs[3].hist2d(
        np.log(y_true_test), 
        np.log(y_pred_test),
        bins=100,
        range=[[low, high], [low, high]],
        cmap=cmap_white_zero,
        norm=colors.LogNorm(vmin=1)  # <-- Log scale for colorbar
    )
    
    # Diagonal reference line
    axs[3].plot([low, high], [low, high], 'k--', linewidth=2)
    
    # Axis labels and formatting
    axs[3].set_xlabel("Ln True Mass [GeV]", fontsize=label_fontsize)
    axs[3].set_ylabel("Ln Predicted Mass [GeV]", fontsize=label_fontsize)
    axs[3].set_title("Ln 2D Histogram [GeV]", fontsize=title_fontsize)
    axs[3].tick_params(axis='both', labelsize=tick_fontsize)
    axs[3].grid(True, linestyle='--', alpha=0.3)
    axs[3].minorticks_on()
    
    # Colorbar with proper labeling
    fig2.colorbar(
        hist_general[3], ax=axs[3], label='Log Counts', pad=0.02, fraction=0.05
    ).ax.tick_params(labelsize=tick_fontsize)
    
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, '2d_hist_by_mass_range.png'))
    plt.close(fig2)

    # === THIRD FIGURE: Three Mass Range Histograms ===
    fig3, axs = plt.subplots(1, 3, figsize=(20, 6))
    
    # --- Panel 1: Coarse bins: 40–80 ---
    bins1 = 20
    range1 = [40, 80]
    bin_width1 = (range1[1] - range1[0]) / bins1
    
    axs[0].hist(y_true_test, bins=bins1, range=range1, histtype='step', color='orange', label='Truth', linewidth=linewidth)
    axs[0].hist(y_pred_test, bins=bins1, range=range1, histtype='step', color='blue', label='Prediction', linewidth=linewidth)
    axs[0].set_xlabel('Mass [GeV]', fontsize=label_fontsize)
    axs[0].set_ylabel('Counts', fontsize=label_fontsize)
    axs[0].set_title(f'Mass Range: 40–80 GeV (Bin width: {bin_width1:.1f} GeV)', fontsize=title_fontsize)
    axs[0].legend(fontsize=legend_fontsize, framealpha=0.8)
    axs[0].grid(True, linestyle='--', alpha=0.3)
    axs[0].tick_params(axis='both', labelsize=tick_fontsize)
    axs[0].minorticks_on()
    
    # --- Panel 2: Fine bins: 80–110 ---
    bins2 = 100
    range2 = [80, 110]
    bin_width2 = (range2[1] - range2[0]) / bins2
    
    axs[1].hist(y_true_test, bins=bins2, range=range2, histtype='step', color='orange', label='Truth', linewidth=linewidth)
    axs[1].hist(y_pred_test, bins=bins2, range=range2, histtype='step', color='blue', label=f'Prediction, MedianAE: {median_ae:.3f}', linewidth=linewidth)
    axs[1].set_xlabel('Mass [GeV]', fontsize=label_fontsize)
    axs[1].set_ylabel('Counts', fontsize=label_fontsize)
    axs[1].set_title(f'Mass Range: 80–110 GeV (Bin width: {bin_width2:.1f} GeV)', fontsize=title_fontsize)
    axs[1].legend(fontsize=legend_fontsize, framealpha=0.8)
    axs[1].grid(True, linestyle='--', alpha=0.3)
    axs[1].tick_params(axis='both', labelsize=tick_fontsize)
    axs[1].minorticks_on()
    
    # --- Panel 3: Coarse bins: 110–1200 ---
    bins3 = 30
    range3 = [110, 1200]
    bin_width3 = (range3[1] - range3[0]) / bins3
    
    axs[2].hist(y_true_test, bins=bins3, range=range3, histtype='step', color='orange', label='Truth', linewidth=linewidth)
    axs[2].hist(y_pred_test, bins=bins3, range=range3, histtype='step', color='blue', label='Prediction', linewidth=linewidth)
    axs[2].set_xlabel('Mass [GeV]', fontsize=label_fontsize)
    axs[2].set_ylabel('Counts', fontsize=label_fontsize)
    axs[2].set_title(f'Mass Range: 110–1200 GeV (Bin width: {bin_width3:.1f} GeV)', fontsize=title_fontsize)
    axs[2].legend(fontsize=legend_fontsize, framealpha=0.8)
    axs[2].grid(True, linestyle='--', alpha=0.3)
    axs[2].tick_params(axis='both', labelsize=tick_fontsize)
    axs[2].minorticks_on()
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, 'mass_histograms_ranges.png'))
    plt.close(fig3)

def extract_layer_mats(attn_per_example):
    """
    Convert a list of attention tensors (for one example) into numpy matrices per layer.
    - attn_per_example: list of torch.Tensor on CPU, each of shape (heads, seq_len, seq_len)
                        or (seq_len, seq_len).
    Returns a list of numpy arrays of shape (seq_len, seq_len), averaged over heads if needed.
    """
    layers = []
    for attn in attn_per_example:
        arr = attn.numpy()  # dims = 3 (heads, seq, seq) or 2 (seq, seq)
        if arr.ndim == 3:
            arr = arr.mean(axis=0)  # average over head dimension
        layers.append(arr)
    return layers


def compute_attention_rollout(layer_mats):
    """
    layer_mats: list of numpy arrays of shape
      • (batch, heads, seq, seq), or
      • (batch, seq, seq), or
      • (heads, seq, seq), or
      • (seq, seq)

    This will average over all leading dims so that each mat is (seq, seq).
    """
    mats = []
    for mat in layer_mats:
        # collapse any extra dims beyond the final two
        extra_axes = tuple(range(mat.ndim - 2))
        if extra_axes:
            mat = mat.mean(axis=extra_axes)
        # now mat.shape == (seq, seq)
        mats.append(mat)

    # sanity check: they all use the same sequence length
    seq_len = mats[0].shape[-1]
    assert all(m.shape == (seq_len, seq_len) for m in mats), \
        "Inconsistent seq_len in attention matrices"

    # build augmented matrices and multiply
    aug = [
        (m + np.eye(seq_len)) /
        (m.sum(axis=-1, keepdims=True) + 1e-6)
        for m in mats
    ]
    rollout = aug[0]
    for m in aug[1:]:
        rollout = m @ rollout
    return rollout


def plot_and_save_attention(attn_per_example, save_dir):
    """
    Given one example’s list of attention tensors and a target directory,
    computes attention rollout and token importance, then saves two plots.

    Args:
        attn_per_example: list of torch.Tensor on CPU, each of shape
                          (heads, seq_len, seq_len) or (seq_len, seq_len).
        save_dir (str): directory path where to save the plots.
    
    Returns:
        dict: paths to the saved images, keys "rollout" and "importance".
    """
    # ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # --- extract per-layer numpy mats ---
    layer_mats = []
    for attn in attn_per_example:
        arr = attn.detach().cpu().numpy()
        # collapse any dimensions beyond the final two (batch, heads, etc.)
        extra_axes = tuple(range(arr.ndim - 2))
        if extra_axes:
            arr = arr.mean(axis=extra_axes)
        # now arr.shape == (seq_len, seq_len)
        layer_mats.append(arr)

    # --- compute rollout ---
    seq_len = layer_mats[0].shape[0]
    aug = [(mat + np.eye(seq_len)) / (mat.sum(axis=-1, keepdims=True) + 1e-6)
           for mat in layer_mats]
    rollout = aug[0]
    for mat in aug[1:]:
        rollout = mat @ rollout

    # Only two jets for now
    num_interaction_features = seq_len - 5
    labels = ["lep", "tau", "MET"] + ["jet1", "jet2"] + [f"jet{i+1}" for i in range(num_interaction_features)]

    # --- plot & save rollout heatmap ---
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    im = ax1.imshow(rollout, aspect='auto')
    
    # set ticks at every position
    ax1.set_xticks(range(seq_len))
    ax1.set_yticks(range(seq_len))
    
    # label them with our token names (rotate x for readability)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_yticklabels(labels)
    
    ax1.set_title('Attention Rollout Heatmap')
    ax1.set_xlabel('Source Token')
    ax1.set_ylabel('Target Token')
    fig1.colorbar(im, ax=ax1, label='Rollout Weight')
    
    rollout_path = os.path.join(save_dir, "rollout.png")
    fig1.tight_layout()
    fig1.savefig(rollout_path)
    plt.close(fig1)
    
    
    # --- compute & plot token importance ---
    importance = rollout.sum(axis=0)
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(range(seq_len), importance)
    
    # apply the same labels
    ax2.set_xticks(range(seq_len))
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    
    ax2.set_title('Token Importance from Rollout')
    ax2.set_xlabel('Token')
    ax2.set_ylabel('Accumulated Importance')
    
    fig2.tight_layout()
    importance_path = os.path.join(save_dir, "importance.png")
    fig2.savefig(importance_path)
    plt.close(fig2)



def permutation_feature_importance_fast(
    model, val_loader, metric_fn, device,      
    higher_is_better=True, transformer=False, batch_size=1024
):
    # 1) Preload entire dataset once onto GPU as a single tensor
    X_list, y_list = [], []
    for Xb, yb in val_loader:
        X_list.append(Xb)
        y_list.append(yb)
    X_val_t = torch.cat(X_list, dim=0).to(device)         # shape (N,...) on GPU
    y_true  = torch.cat(y_list, dim=0).to(device).squeeze()

    N = X_val_t.shape[0]
    if transformer:
        _, T, F = X_val_t.shape
        importances = torch.zeros(T, F, device=device)
    else:
        _, F = X_val_t.shape
        importances = torch.zeros(F, device=device)
        
    # 2) Compute baseline prediction once
    with torch.no_grad():
        preds_base = model(X_val_t).squeeze()
    base_score = metric_fn(y_true.cpu().numpy(), preds_base.cpu().numpy())

    # 3) Helper to run batched prediction on a GPU tensor
    def batched_predict(X):
        out = []
        with torch.no_grad():
            for i in range(0, N, batch_size):
                out.append(model(X[i : i + batch_size]))
        return torch.cat(out).squeeze()

    # 4) In‐place shuffle + restore for each feature (or token×feature)
    if transformer:
        for t in range(T):
            for f in range(F):
                # backup column t,f
                backup = X_val_t[:, t, f].clone()
                perm    = torch.randperm(N, device=device)
                X_val_t[:, t, f] = X_val_t[perm, t, f]

                preds_p = batched_predict(X_val_t)
                score_p = metric_fn(y_true.cpu().numpy(), preds_p.cpu().numpy())
                delta   = (base_score - score_p) if higher_is_better else (score_p - base_score)
                importances[t, f] = delta

                # restore
                X_val_t[:, t, f] = backup

    else:
        for f in range(F):
            backup = X_val_t[:, f].clone()
            perm   = torch.randperm(N, device=device)
            X_val_t[:, f] = X_val_t[perm, f]

            preds_p = batched_predict(X_val_t)
            score_p = metric_fn(y_true.cpu().numpy(), preds_p.cpu().numpy())
            delta   = (base_score - score_p) if higher_is_better else (score_p - base_score)
            importances[f] = delta

            X_val_t[:, f] = backup

    return importances.cpu().numpy()

def plot_feature_importance(
    importances, result_dir, feature_names, transformer=False
):
    os.makedirs(result_dir, exist_ok=True)

    logger.info("Now plotting feature importance")
    if transformer:
        # Token × Feature heatmap
        T, F = importances.shape
        
        logger.info(f'Importances shape: {importances.shape}')
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            importances,
            xticklabels=feature_names,
            yticklabels=[f"token {i}" for i in range(T)],
            annot=True, fmt=".3f", ax=ax
        )
        ax.set_title("Permutation Importance per Token × Feature")
        plt.tight_layout()
        plt.savefig(f"{result_dir}/token_importance.png", dpi=150)
        plt.close(fig)
    else:
        # Simple bar chart for non-transformer
        F = importances.shape[0]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(feature_names, importances, color='steelblue')
        ax.set_xlabel("Importance (Δ score)")
        ax.set_title("Feature Importances")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{result_dir}/feature_importance.png", dpi=150)
        plt.close(fig)

# Example of integrating in compute_feature_importance_and_correlation_plot
def compute_feature_importance_and_correlation_plot(
    model, datamodule, device, result_dir,
    feature_names, trainer=None, transformer=False
):
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

    # In transformer mode override feature names
    if transformer:
        _, T, F = X_val.shape
        feature_names = ['energy', 'eta', 'cos(phi)', 'sin(phi)', 'pt', 'btag', 'charge', 'type']
    
    # Compute permutation importances for each metric
    imp_mse = permutation_feature_importance_fast(
        model, val_loader, mean_squared_error, device, higher_is_better=False,
        transformer=transformer
    )
    imp_r2 = permutation_feature_importance_fast(
        model, val_loader, r2_score, device, higher_is_better=True,
        transformer=transformer
    )
    imp_mae = permutation_feature_importance_fast(
        model, val_loader, mean_absolute_error, device, higher_is_better=False,
        transformer=transformer
    )

    os.makedirs(result_dir, exist_ok=True)
    if transformer:
        # Token × Feature heatmap for transformer models

        T, F = imp_mse.shape
        yticklabels = ["Lep", "Tau", "MET"] + [f"Jet {i}" for i in range(T - 3)]

        for metric_name, importance_matrix in zip(
            ["mse", "mae", "r2"],
            [imp_mse, imp_mae, imp_r2]
        ):
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(
                importance_matrix,
                xticklabels=feature_names,
                yticklabels=yticklabels,
                annot=True, fmt=".3f", ax=ax
            )
            ax.set_title(f"Permutation Importance per Token × Feature ({metric_name.upper()})")
            plt.tight_layout()
            plt.savefig(f"{result_dir}/token_feature_importance_{metric_name}.png", dpi=150)
            plt.close(fig)

    else:
        # Simple bar chart for non-transformer models
        for metric_name, importance_vector, xlabel in zip(
            ["mse", "mae", "r2"],
            [imp_mse, imp_mae, imp_r2],
            [
                "Increase in MSE after permutation",
                "Increase in MAE after permutation",
                "Decrease in R² after permutation"
            ]
        ):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(feature_names, importance_vector, color='steelblue')
            ax.set_xlabel(xlabel)
            ax.set_title(f"Feature Importances ({metric_name.upper()})")
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"{result_dir}/feature_importance_{metric_name}.png", dpi=150)
            plt.close(fig)


def plot_resampled_distributions(y_train, y_train_inverse_sampled, save_path):
    # Create directory if needed
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Define zoomed-in regions
    regions = [(40, 80), (80, 110), (110, 1500)]
    region_titles = ["Low Range (40–80) GeV", "Mid Range (80–110) GeV", "High Range (110–1500) GeV"]

    # Set up compact layout
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    logger.info(len(y_train))
    logger.info(len(y_train_inverse_sampled))
    for ax, (low, high), title in zip(axes, regions, region_titles):
        ax.hist(
            y_train, bins=50, range=(low, high), alpha=0.6, label='Original', 
            color='steelblue', density=True
        )
        ax.hist(
            y_train_inverse_sampled, bins=50, range=(low, high), alpha=0.6, label='Resampled', 
            color='darkorange', density=True
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Target Value')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    axes[0].set_ylabel('Density')
    axes[1].legend(loc='upper right', fontsize=10)
    
    plt.savefig(save_path, dpi=300)


def plot_kde_and_inverse_weights(
    y_train,
    density,
    inv_density,
    save_path=None,
    xlim=(60, 120)
):
    """
    Plot the KDE density and inverse sampling weights against the target values.

    Args:
        y_train (np.ndarray): The original target values.
        density (np.ndarray): KDE density values for each y_train.
        inv_density (np.ndarray): Inverse density values (weights).
        save_path (str or None): Path to save the plot. If None, just shows.
        xlim (tuple): X-axis limits (min, max) for the plot.
    """
    # Sort for smooth curves
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
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

