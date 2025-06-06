"""
Custom trainer implementations.
"""

from transformers.trainer import Trainer

from lmv3.losses import FocalLoss, FocalLossAlpha, FocalLossGamma, TaskType


class FocalLossTrainer(Trainer):
    """
    Custom Trainer that uses Focal Loss instead of standard CrossEntropyLoss.

    Useful for handling class imbalance in token classification tasks.
    """

    def __init__(
        self,
        *args,
        focal_loss_alpha: FocalLossAlpha = 1.0,
        focal_loss_gamma: FocalLossGamma = 2.0,
        task_type: TaskType,
        num_classes: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Initialize focal loss with provided parameters
        self.focal_loss = FocalLoss(
            gamma=focal_loss_gamma,
            alpha=focal_loss_alpha,
            task_type=task_type,
            num_classes=num_classes,
        )
        print(
            f"Initialized FocalLoss with alpha={focal_loss_alpha}, gamma={focal_loss_gamma}"
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Override compute_loss to use Focal Loss instead of default CrossEntropyLoss.
        """
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Reshape for token classification
        # logits: (batch_size, sequence_length, num_labels)
        # labels: (batch_size, sequence_length)

        if labels is not None:
            # Flatten the tokens
            active_loss = labels.view(-1) != -100
            active_logits = logits.view(-1, self.model.config.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]

            # Compute focal loss
            loss = self.focal_loss(active_logits, active_labels)

            # Handle gradient accumulation scaling if needed
            # The Trainer will automatically handle this, but you can be explicit:
            # if num_items_in_batch is not None:
            #     loss = loss / num_items_in_batch

        else:
            loss = None

        return (loss, outputs) if return_outputs else loss
