
# Configuration Files

This README explains how to use the YAML configuration file to set up dataset loading, model architecture, training, and prediction parameters for the Transformer regression model.

Example config files can be found in the `tests` folder.

---

## Dataset Configuration

- **`input_files`**  
  Expects an array of `.h5` files.

- **`scaling_dict`**  
  A dictionary specifying how each variable is scaled. Available scalers include:  
  `log`, `logminmax`, `minmax`, `standard`, `arctan`, `tanh`, and custom physical scalers.  
  More details on scalers can be found in [`src/data/DataScalers.py`](src/data/DataScalers.py).  
  All data before and after scaling is plotted.  
  Use `"none"` to apply no scaling. Variables not listed will be excluded from the model.

- **`target_scaler`**  
  Specifies how to scale the target variable; plots are also generated.

- **`train` and `val`**  
  - `batch_size`: Number of events per batch  
  - `num_workers`: Number of CPU workers used for data loading

- **`inverse_sampling`**: If `True`, samples batches unevenly to prioritize rare events by building a KDE with specified parameters and using its inverse distribution.  
- **`KDE_width`**: Width parameter for Gaussian KDE.  
- **`Min_Dens_cap`**: Caps minimum density to prevent excessively large sampling weights for ultra-rare events.

---

## Model Configuration

- **`type`**  
  Specify model type: `DNNRegression` or `TransformerRegression`.

- **`name`**  
  String identifying the model. This becomes the name of the results directory containing all generated plots.

- **`n_particles`**  
  Number of particles in the data, ordered as (lep, tau, MET, jet1, jet2, jet3, etc.).  
  The model adjusts automatically if more than 5 particles are used and applies appropriate masking.

### DNN-specific Parameters:

- **`layers`**  
  Array specifying hidden layer sizes.  
  Example: `[128, 128]` means two hidden layers with 128 neurons each.

### Transformer-specific Parameters:

- **`num_heads`**  
  Number of attention heads.

- **`num_layers`**  
  Number of stacked transformer layers.

- **`input_embedder_NN_layers`**  
  Array defining neuron counts in the input embedder layers. The last value is the embedding dimension and must be divisible by the number of heads.

- **`output_activation`**  
  Activation function for output neuron: `softplus`, `sigmoid`, `tanh`, `relu`, or `linear` (no activation).

- **`pooling_type`**  
  Either `mean` or `attention`.

- **`norm_type`**  
  Normalization method between input embedder layers: `none`, `batchnorm`, or `layernorm`.

---

## Training Configuration

- **`dropout_probability`**  
  Dropout rate applied to hidden layers.

- **`early_stopping_patience`**  
  Number of epochs with no improvement before stopping training.

- **`learning_rate`**  
  Learning rate for optimizer.

- **`loss_fn`**  
  Loss function name.

- **`loss_fn_params`**  
  Dictionary of parameters to configure the loss function.  
  Supports nested loss functions, e.g.:

```yaml
loss_fn: QuantileAwareLoss
loss_fn_params:
  alpha: 0.1
  squared: false
  base_loss: SmoothL1Loss
  base_loss_params:
    beta: 10.0
    reduction: mean
```

Available loss functions are listed in [`src/utils/LossFunctions.py`](src/utils/LossFunctions.py).

- **`lr_scheduler_patience`**  
  Number of epochs to wait before reducing learning rate when validation loss plateaus (default scheduler: `PlateauReduceLR`).

- **`n_epochs`**  
  Total number of training epochs.

- **`num_quantiles`**  
  Number of quantiles used when employing `QuantileAwareLoss`. Ignored otherwise.

- **`weight_decay`**  
  L2 regularization strength. Set to `0` to disable.

- **`optimize`**  
  Enables PyTorch TF32 matmul optimization (trades precision for speed).

- **Transformer-specific:**  
  - `compute_interaction_tokens`: Set to `True` to add interaction tokens between particles as described in [10.1088/1674-1137/ad7f3d]. Adds a separate interaction embedder with layers matching the particle embedder.

---

## General Commands

- **`device`**  
  e.g., `cuda` or `cpu`.

- **`mode`**  
  Model mode: `train`, `performance`, or `predict`.

- **`runner`**  
  Identifier for the runner (e.g., `Bob`).

---

## Output & Metrics

- Statistical such as: MeanAbsoluteError, MeanSquaredError, R2Score, MeanFractionalBias, MedianAbsoluteError and RootMeanSquaredLogError are calculated for the DNN and the Transformer.
- Extra attention-specific metrics are computed for the Transformer: AttentionEntropy, AttentionSparsity, HeadDiversity

- Feature importance plots and performance-related plots are generated and saved in the model directory.

- When KDE sampling is enabled, plots comparing original vs. new sampling distributions are included.
- 
---
