# In this file model metrics which are not available in the torchmetrics library are defined
import torch
import torch.nn.functional as F
import numpy as np
import torchmetrics


class MeanFractionalBias(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum_fractional_bias", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Convert to numpy for calculation if needed
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        # Calculate mean fractional bias
        fractional_bias = 2 * (preds - target) / (preds + target)
        # Handle division by zero
        mask = (preds + target) != 0
        fractional_bias = fractional_bias[mask]
        
        if len(fractional_bias) > 0:
            mfb = np.mean(fractional_bias)
            self.sum_fractional_bias += torch.tensor(mfb * len(fractional_bias))
            self.total += len(fractional_bias)

    def compute(self):
        return self.sum_fractional_bias / self.total


class MedianAbsoluteError(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("errors", default=torch.tensor([]), dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        abs_errors = torch.abs(preds - target).flatten()
        # Concatenate with existing errors instead of using a list
        if self.errors.numel() == 0:
            self.errors = abs_errors
        else:
            self.errors = torch.cat([self.errors, abs_errors], dim=0)

    def compute(self):
        if self.errors.numel() == 0:
            return torch.tensor(0.0)
        return torch.median(self.errors)


class RootMeanSquaredLogError(torchmetrics.Metric):
    def __init__(self):
        super().__init__()

        # Store squared log errors
        self.add_state("squared_log_errors", default=torch.tensor([]), dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Clamp to avoid negative values (RMSLE requires non-negative inputs)
        preds = torch.clamp(preds, min=0)
        target = torch.clamp(target, min=0)

        # Compute squared log errors
        log_preds = torch.log1p(preds)
        log_target = torch.log1p(target)
        sq_log_errors = torch.pow(log_preds - log_target, 2).flatten()

        if self.squared_log_errors.numel() == 0:
            self.squared_log_errors = sq_log_errors
        else:
            self.squared_log_errors = torch.cat([self.squared_log_errors, sq_log_errors], dim=0)

    def compute(self):
        if self.squared_log_errors.numel() == 0:
            return torch.tensor(0.0)

        return torch.sqrt(torch.mean(self.squared_log_errors))


# Transformer specific metrics
class AttentionEntropy(torchmetrics.Metric):
    """
    Computes the average entropy of the attention distributions.
    """
    def __init__(self):
        super().__init__()
        self.add_state("entropies", default=torch.tensor([]), dist_reduce_fx="cat")

    def update(self, attn_weights: torch.Tensor):
        eps = 1e-9
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=-1)  # [batch, heads, seq_len]
        avg_entropy = entropy.mean(dim=(-1, -2))  # average per sample
        if self.entropies.numel() == 0:
            self.entropies = avg_entropy.flatten()
        else:
            self.entropies = torch.cat([self.entropies, avg_entropy.flatten()], dim=0)

    def compute(self):
        if self.entropies.numel() == 0:
            return torch.tensor(0.0)
        return torch.mean(self.entropies)


class AttentionSparsity(torchmetrics.Metric):
    """
    Computes the proportion of attention weights above a given threshold.
    """
    def __init__(self, threshold: float = 0.05):
        super().__init__()
        self.threshold = threshold
        self.add_state("sparsities", default=torch.tensor([]), dist_reduce_fx="cat")

    def update(self, attn_weights: torch.Tensor):
        sparse_mask = attn_weights > self.threshold
        sparsity_ratio = sparse_mask.float().mean(dim=(-1, -2, -3))  # per sample
        if self.sparsities.numel() == 0:
            self.sparsities = sparsity_ratio.flatten()
        else:
            self.sparsities = torch.cat([self.sparsities, sparsity_ratio.flatten()], dim=0)

    def compute(self):
        if self.sparsities.numel() == 0:
            return torch.tensor(0.0)
        return torch.mean(self.sparsities)


class HeadDiversity(torchmetrics.Metric):
    """
    Computes how diverse the attention heads are within a batch with Cosine similarity.
    """
    def __init__(self):
        super().__init__()
        self.add_state("diversities", default=torch.tensor([]), dist_reduce_fx="cat")

    def update(self, attn_weights: torch.Tensor):
        batch_size, num_heads, seq_len, _ = attn_weights.shape
        scores = []
        for i in range(batch_size):
            for j in range(num_heads):
                for k in range(j + 1, num_heads):
                    a = attn_weights[i, j].flatten()
                    b = attn_weights[i, k].flatten()
                    sim = F.cosine_similarity(a, b, dim=0)
                    scores.append(1.0 - sim)
        if len(scores) > 0:
            div_score = torch.stack(scores).mean()
            if self.diversities.numel() == 0:
                self.diversities = div_score.unsqueeze(0)
            else:
                self.diversities = torch.cat([self.diversities, div_score.unsqueeze(0)], dim=0)

    def compute(self):
        if self.diversities.numel() == 0:
            return torch.tensor(0.0)
        return torch.mean(self.diversities)