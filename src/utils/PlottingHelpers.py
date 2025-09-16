# In this file we define functions called often in the plotting routines of the code, these functions range from 
# computing grids and masks to calling the correct matplotlib functions to stylize the plots
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import torch
from sklearn.metrics import r2_score
from pathlib import Path

def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_fig(fig, path, dpi=150):
    _ensure_dir(Path(path).parent)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path

    
def compute_loss_landscape(criterion):
    """Compute loss landscape for 3D visualization."""
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
    
    return loss_grid, P.numpy(), T.numpy()


def get_colormap_with_white_zero():
    """Create colormap with white background for zero values."""
    cmap = plt.cm.viridis.copy()
    cmap.set_under('white')
    return cmap


def style_axis(ax, xlabel, ylabel, title, style):
    """Apply consistent styling to an axis."""
    ax.set_xlabel(xlabel, fontsize=style['label_fontsize'])
    ax.set_ylabel(ylabel, fontsize=style['label_fontsize'])
    ax.set_title(title, fontsize=style['title_fontsize'])
    ax.legend(fontsize=style['legend_fontsize'], framealpha=0.8)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', labelsize=style['tick_fontsize'])
    ax.minorticks_on()