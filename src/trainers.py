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
    
    def __init__(self, *args, focal_loss_alpha=1, focal_loss_gamma=2, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize focal loss with provided parameters
        self.focal_loss = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma)
        print(f"Initialized FocalLoss with alpha={focal_loss_alpha}, gamma={focal_loss_gamma}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute focal loss instead of standard cross entropy.
        
        Args:
            model: The model to train
            inputs: Input batch including labels
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (unused)
            
        Returns:
            Loss value and optionally model outputs
        """
        # Extract labels
        labels = inputs.pop("labels")

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Filter out ignored indices (-100)
        active_loss = labels.view(-1) != -100
        active_logits = logits.view(-1, self.model.config.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]

        if active_labels.numel() == 0:  # Handle cases with no valid labels in batch
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            loss = self.focal_loss(active_logits, active_labels)
            
            # Debug: log label distribution in batch
            if hasattr(self, 'step_count'):
                self.step_count += 1
            else:
                self.step_count = 1
                
            if self.step_count % 50 == 0:  # Log every 50 steps
                unique_labels, counts = torch.unique(active_labels, return_counts=True)
                print(f"Step {self.step_count}: Label distribution - {dict(zip(unique_labels.cpu().tolist(), counts.cpu().tolist()))}")
                print(f"Loss: {loss.item():.4f}")

        return (loss, outputs) if return_outputs else loss