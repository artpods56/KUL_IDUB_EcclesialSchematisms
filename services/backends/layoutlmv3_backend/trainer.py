import torch
from losses import FocalLoss
from transformers.trainer import Trainer


class FocalLossTrainer(Trainer):
    def __init__(self, *args, focal_loss_alpha=1, focal_loss_gamma=2, **kwargs):
        super().__init__(*args, **kwargs)
        # You can pass alpha/gamma during Trainer instantiation
        # If you want per-class alpha, pass the weight tensor here
        self.focal_loss = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # Extract labels
        labels = inputs.pop("labels")

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Simpler approach: Let FocalLoss compute on all, rely on large number of tokens
        # More correct: Filter ignored indices
        active_loss = labels.view(-1) != -100
        active_logits = logits.view(-1, self.model.config.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]

        if active_labels.numel() == 0:  # Handle cases with no valid labels in batch
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            loss = self.focal_loss(active_logits, active_labels)

        return (loss, outputs) if return_outputs else loss
