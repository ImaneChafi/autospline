"""Loss functions for heatmap regression."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalMSELoss(nn.Module):
    """
    Weighted MSE that upweights positive (keypoint) regions.

    Combines:
      - MSE on positive pixels (high GT heatmap value)
      - Down-weighted MSE on background pixels
    """

    def __init__(self, pos_weight: float = 10.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = 1.0 + (self.pos_weight - 1.0) * target
        return torch.mean(weights * (pred - target) ** 2)


class BinaryFocalLoss(nn.Module):
    """
    Focal loss for binary heatmap prediction (adapted from CornerNet).

    Reduces loss for easy background, focuses on hard examples and positives.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        pred = pred.clamp(eps, 1 - eps)

        pos_mask = (target == 1).float()
        neg_mask = 1 - pos_mask

        pos_loss = -((1 - pred) ** self.alpha) * torch.log(pred) * pos_mask
        neg_loss = -((1 - target) ** self.beta) * (pred ** self.alpha) * torch.log(1 - pred) * neg_mask

        n_pos = pos_mask.sum().clamp(min=1)
        return (pos_loss.sum() + neg_loss.sum()) / n_pos


class CombinedHeatmapLoss(nn.Module):
    """Weighted sum of FocalMSE + SSIM-inspired smoothness term."""

    def __init__(self, focal_weight: float = 1.0, smooth_weight: float = 0.1):
        super().__init__()
        self.focal = FocalMSELoss(pos_weight=20.0)
        self.focal_weight = focal_weight
        self.smooth_weight = smooth_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.focal_weight * self.focal(pred, target)
        if self.smooth_weight > 0:
            # Gradient smoothness: penalise sharp edges in predictions
            dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            loss += self.smooth_weight * (dx.abs().mean() + dy.abs().mean())
        return loss
