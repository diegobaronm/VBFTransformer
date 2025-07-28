# === Standard Library ===
import os

# === Third-Party Libraries ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import seaborn as sns
from sklearn.metrics import r2_score

# === PyTorch and Related ===
import torch
from torch.utils.data import DataLoader


def plot_particle_distributions(peak_indices, signal_data, x_range=None, title='', titles=None, n_bins=40, no_particle_name=False, folder_name=''):
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

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error, r2_score

def permutation_feature_importance(model, val_loader, baseline_metric, device, metric_fn, higher_is_better=True):
    """
    General permutation feature importance for any metric.

    Args:
        model: Trained model
        val_loader: validation DataLoader
        baseline_metric: baseline metric value on unshuffled data
        device: torch device
        metric_fn: function(y_true, y_pred) -> scalar metric
        higher_is_better: bool, True if metric is score (R²), False if loss (MSE)

    Returns:
        importances: np.array of feature importances
    """
    model.eval()
    X_val, y_val = [], []
    for inputs, targets in val_loader:
        X_val.append(inputs.cpu().numpy())
        y_val.append(targets.cpu().numpy())
    X_val = np.concatenate(X_val)
    y_val = np.concatenate(y_val)

    importances = []
    batch_size = val_loader.batch_size if hasattr(val_loader, "batch_size") else 32

    for col in range(X_val.shape[1]):
        X_val_permuted = X_val.copy()
        np.random.shuffle(X_val_permuted[:, col])

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                inputs = torch.tensor(X_val_permuted[i:i+batch_size], dtype=torch.float32).to(device)
                targets = torch.tensor(y_val[i:i+batch_size], dtype=torch.float32).to(device)
                outputs = model(inputs)
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        y_pred_perm = np.concatenate(all_preds).squeeze()
        y_true = np.concatenate(all_targets).squeeze()

        permuted_metric = metric_fn(y_true, y_pred_perm)

        if higher_is_better:
            importance = baseline_metric - permuted_metric  # drop in score
        else:
            importance = permuted_metric - baseline_metric  # increase in loss

        importances.append(importance)

    return np.array(importances)


def compute_feature_importance_and_correlation_plot(model, datamodule, device, result_dir, feature_names, trainer=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    val_loader = trainer.datamodule.val_dataloader() if trainer else datamodule.val_dataloader()
    model.eval()

    # Collect all val inputs and targets once
    X_val, y_val = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            X_val.append(inputs.cpu().numpy())
            y_val.append(targets.cpu().numpy())
    X_val = np.concatenate(X_val)
    y_val = np.concatenate(y_val)

    # Predict on validation set
    with torch.no_grad():
        inputs_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        outputs_t = model(inputs_t).cpu().numpy().squeeze()

    # Compute baseline metrics
    baseline_mse = mean_squared_error(y_val, outputs_t)
    baseline_r2 = r2_score(y_val, outputs_t)

    # Compute permutation importances for MSE and R2
    importances_mse = permutation_feature_importance(
        model, val_loader, baseline_mse, device, mean_squared_error, higher_is_better=False
    )
    importances_r2 = permutation_feature_importance(
        model, val_loader, baseline_r2, device, r2_score, higher_is_better=True
    )

    # Sort and filter top features by MSE importance
    sorted_idx = np.argsort(importances_mse)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    importances_mse_sorted = importances_mse[sorted_idx]
    importances_r2_sorted = importances_r2[sorted_idx]

    # Filter zero or negative importance (can happen if permuted_metric < baseline)
    mask = importances_mse_sorted > 0
    sorted_features = [f for i, f in enumerate(sorted_features) if mask[i]]
    importances_mse_sorted = importances_mse_sorted[mask]
    importances_r2_sorted = importances_r2_sorted[mask]

    # Keep top 15
    top_k = 15
    sorted_features = sorted_features[:top_k]
    importances_mse_sorted = importances_mse_sorted[:top_k]
    importances_r2_sorted = importances_r2_sorted[:top_k]

    # Correlation matrix for top features
    df = pd.DataFrame(X_val, columns=feature_names)
    df_top = df[sorted_features]
    corr_matrix = df_top.corr(method='pearson')

    # Plotting
    height_per_feature = 0.5
    fig_height = max(8, top_k * height_per_feature)
    fig, axes = plt.subplots(1, 3, figsize=(24, fig_height))

    # MSE Importances plot
    axes[0].barh(sorted_features, importances_mse_sorted, color='steelblue')
    axes[0].set_xlabel("Increase in MSE after permutation", fontsize=14)
    axes[0].set_title("Top 15 Feature Importances (MSE)", fontsize=16)
    axes[0].invert_yaxis()
    axes[0].tick_params(axis='y', labelsize=12)
    axes[0].tick_params(axis='x', labelsize=12)

    # R2 Importances plot
    axes[1].barh(sorted_features, importances_r2_sorted, color='darkorange')
    axes[1].set_xlabel("Decrease in R² after permutation", fontsize=14)
    axes[1].set_title("Top 15 Feature Importances (R²)", fontsize=16)
    axes[1].invert_yaxis()
    axes[1].tick_params(axis='y', labelsize=12)
    axes[1].tick_params(axis='x', labelsize=12)

    # Correlation heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        fmt=".2f",
        cmap='coolwarm',
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=axes[2],
        annot=True,
        annot_kws={"size": 10}
    )
    axes[2].set_title("Correlation (Top 15 Features)", fontsize=16)
    axes[2].tick_params(axis='x', labelrotation=90, labelsize=12)
    axes[2].tick_params(axis='y', labelrotation=0, labelsize=12)

    plt.tight_layout()
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(f"{result_dir}/Importance_and_Correlation_MSE_R2.png", dpi=150)
    plt.close(fig)
