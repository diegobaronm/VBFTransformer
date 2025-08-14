import torch
import torch.nn as nn
from loguru import logger
from omegaconf import OmegaConf

# Foward declaration, populated at the bottom of the file
loss_function_dictionary = {}

# Function to build the loss function dynamically based on the parameters from the 
# config.yaml and in the case of quantile-aware losses, the quantiles from the 
# training data only (to avoid leakage from test data)
def build_loss_function(name: str, params: dict, quantiles):
    logger.info(f"Building loss function: {name}")

    if name not in loss_function_dictionary:
        raise ValueError(f"Unknown loss function: {name}")
    
    # Convert DictConfig to mutable dict
    params = OmegaConf.to_container(params, resolve=True)
    logger.info(f"Loss parameters (converted): {params}")

    loss_cls = loss_function_dictionary[name]

    # If QuantileAwareLoss, inject quantiles
    if name == 'QuantileAwareLoss':
        params['quantile_edges'] = quantiles
        logger.info(f"Injected quantile_edges into {name}")

    # If base_loss is present, resolve it first
    if 'base_loss' in params:
        base_loss_name = params.pop('base_loss')
        base_loss_params = params.pop('base_loss_params', {})

        logger.info(f"Using base loss: {base_loss_name}")
        logger.info(f"Base loss parameters: {base_loss_params}")

        base_loss_cls = loss_function_dictionary.get(base_loss_name)
        if base_loss_cls is None:
            raise ValueError(f"Unknown base loss: {base_loss_name}")

        base_loss = base_loss_cls(**base_loss_params)
        full_loss = loss_cls(base_loss=base_loss, **params)

        logger.info(f"Constructed compound loss function: {full_loss}")
        return full_loss

    # Regular loss
    full_loss = loss_cls(**params)
    logger.info(f"Constructed loss function: {full_loss}")
    return full_loss


class WeightedTailLoss(nn.Module):
    def __init__(self, base_loss=nn.SmoothL1Loss(beta=10, reduction='mean'), mass_weight=[2.0, 2.0], threshold=[85.0, 100.0]):
        super().__init__()
        self.base_loss = base_loss
        self.mass_weight = mass_weight  # [left_tail_weight, right_tail_weight]
        self.threshold = threshold      # [left_threshold, right_threshold]

        # When weighting loss function results the reduction step needs to be done post weighting
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


# In this loss function, the higher alpha is, the more the loss from the quantiles matters, notably torch.bucketize is a step function 
# hence high values of alpha lead to more unstable gradient-descent motivated training as the gradients are not well defined
class QuantileAwareLoss(nn.Module):
    def __init__(self, quantile_edges=[], base_loss=nn.SmoothL1Loss(beta=10, reduction='mean'), alpha=0.5, squared=False):
        super().__init__()
        
        if len(quantile_edges) < 2:
            raise ValueError("quantile_edges must contain at least 2 values to define 1 bin")
   
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

class CauchyLoss(nn.Module):
    def __init__(self, c=5.0, reduction='mean'):
        """
        Cauchy loss function: log(1 + (diff/c)^2)
        
        Args:
            c (float): Scale parameter controlling outlier sensitivity.
                       Smaller c = more sensitive to errors (use near peak).
                       Larger c = more outlier robustness.
            reduction (str): 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.c = c
        self.reduction = reduction

    def forward(self, input, target):
        diff = input - target
        loss = torch.log1p((diff / self.c)**2)  # log1p for numerical stability
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

            
class InverseGaussianWeightedLoss(nn.Module):
    def __init__(self, base_loss=nn.SmoothL1Loss(beta=10, reduction='mean'), center=91.0, sigma=5.0, max_weight=10.0):
        super().__init__()
        self.base_loss = base_loss
        self.center = center
        self.sigma = sigma
        self.max_weight = max_weight

        # When weighting loss function results the reduction step needs to be done post weighting
        if hasattr(self.base_loss, 'reduction'):
            self.base_loss.reduction = 'none' 

    def forward(self, input, target):
        input = input.squeeze()
        target = target.squeeze()

        loss = self.base_loss(input, target)

        gaussian = torch.exp(-0.5 * ((target - self.center) / self.sigma) ** 2)
        weights = 1.0 / (gaussian + 1e-8)  # inverse Gaussian

        # Cap weights so tails don't get exponentially over-represented
        weights = torch.clamp(weights, max=self.max_weight)

        weighted_loss = loss * weights
        return weighted_loss.mean()

loss_function_dictionary.update({
    'L1Loss': nn.L1Loss,
    'MSELoss': nn.MSELoss,
    'SmoothL1Loss': nn.SmoothL1Loss,
    'WeightedTailLoss': WeightedTailLoss,
    'QuantileAwareLoss': QuantileAwareLoss,
    'InverseGaussianWeightedLoss': InverseGaussianWeightedLoss,
    'CauchyLoss' : CauchyLoss,
})