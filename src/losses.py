from torch import nn
import torch
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'): # Added reduction parameter
        super().__init__()
        # alpha can be a scalar or a tensor of weights per class
        self.alpha = alpha
        self.gamma = gamma
        # reduction can be 'mean', 'sum', or 'none'
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: Logits (N, C)
        # targets: Ground truth labels (N)

        # Calculate Cross Entropy loss without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=-100)

        # Calculate pt (probability of the true class)
        pt = torch.exp(-ce_loss)

        # Calculate Focal Loss
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        # Apply alpha weighting
        if isinstance(self.alpha, torch.Tensor):
            # Ensure alpha is on the same device and apply per-class weight
            alpha_t = self.alpha.to(inputs.device)[targets]
            loss = alpha_t * loss
        elif isinstance(self.alpha, (int, float)) and self.alpha != 1:
             # Apply scalar alpha
            loss = self.alpha * loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none'
            return loss