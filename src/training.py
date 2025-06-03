"""
Main training module that orchestrates the training process.
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers.data.data_collator import default_data_collator
from transformers import Trainer
import torch
from typing import Optional, Dict, Any

from .config import TrainingConfig
from .trainers import FocalLossTrainer, LabelSmoothingTrainer, HybridLossTrainer
from .losses import FocalLoss, LabelSmoothingLoss


def create_trainer(
    config: TrainingConfig,
    model,
    train_dataset,
    eval_dataset,
    compute_metrics,
    tokenizer=None
) -> Trainer:
    """
    Create the appropriate trainer based on configuration.
    
    Args:
        config: Training configuration
        model: The model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        compute_metrics: Function to compute metrics
        tokenizer: Tokenizer (optional)
        
    Returns:
        Configured trainer instance
    """
    training_args = config.to_training_arguments()
    loss_params = config.get_loss_params()
    
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": default_data_collator,
        "compute_metrics": compute_metrics,
        "tokenizer": tokenizer,
    }
    
    if config.loss_type == "focal":
        return FocalLossTrainer(**trainer_kwargs, **loss_params)
    elif config.loss_type == "smoothing":
        return LabelSmoothingTrainer(**trainer_kwargs, **loss_params)
    elif config.loss_type == "hybrid":
        return HybridLossTrainer(**trainer_kwargs, **loss_params)
    else:  # standard
        return Trainer(**trainer_kwargs)


def setup_model_and_tokenizer(
    model_name: str,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    device: Optional[str] = None
):
    """
    Setup model and tokenizer for training.
    
    Args:
        model_name: Name or path of the model
        id2label: Mapping from label IDs to label names
        label2id: Mapping from label names to label IDs
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with label mappings
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    if device:
        model.to(device)
    
    return model, tokenizer


def train_model(
    config: TrainingConfig,
    train_dataset,
    eval_dataset,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    compute_metrics,
    device: Optional[str] = None
):
    """
    Complete training pipeline.
    
    Args:
        config: Training configuration
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        id2label: Mapping from label IDs to label names
        label2id: Mapping from label names to label IDs
        compute_metrics: Function to compute metrics
        device: Device to train on
        
    Returns:
        Trained model and trainer
    """
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        config.model_name, id2label, label2id, device
    )
    
    # Create trainer
    trainer = create_trainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    # Train the model
    print(f"Starting training with {config.loss_type} loss...")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    
    return model, trainer


def load_id2label_mappings(id2label_path: str, label2id_path: str):
    """
    Load label mappings from JSON files.
    
    Args:
        id2label_path: Path to id2label JSON file
        label2id_path: Path to label2id JSON file
        
    Returns:
        Tuple of (id2label, label2id) dictionaries
    """
    import json
    
    with open(id2label_path, 'r') as f:
        id2label = json.load(f)
    
    with open(label2id_path, 'r') as f:
        label2id = json.load(f)
    
    # Convert id2label keys to integers
    id2label = {int(k): v for k, v in id2label.items()}
    
    return id2label, label2id


def get_device():
    """Get the best available device for training."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"