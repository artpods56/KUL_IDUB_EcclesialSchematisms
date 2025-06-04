"""
Custom trainer implementations.
"""

from transformers import Trainer
import torch
from losses import FocalLoss


class FocalLossTrainer(Trainer):
    """
    Custom Trainer that uses Focal Loss instead of standard CrossEntropyLoss.
    
    Useful for handling class imbalance in token classification tasks.
    """
    
    def __init__(self, *args, focal_loss_alpha=1, focal_loss_gamma=2, task_type: str, num_classes: int, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize focal loss with provided parameters
        self.focal_loss = FocalLoss(gamma=focal_loss_gamma, alpha=focal_loss_alpha, task_type=task_type, num_classes=num_classes)
        print(f"Initialized FocalLoss with alpha={focal_loss_alpha}, gamma={focal_loss_gamma}")