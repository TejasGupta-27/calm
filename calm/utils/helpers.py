"""
Helper utilities for CALM implementation.

Includes logging, checkpointing, and general utilities.
"""

import os
import json
import random
from pathlib import Path
import logging

import torch
import numpy as np


def setup_logging(log_dir, log_file='train.log'):
    """
    Setup logging configuration.

    Args:
        log_dir (str): Directory to save log files
        log_file (str): Name of log file

    Returns:
        logging.Logger: Configured logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / log_file

    # Create logger
    logger = logging.getLogger('CALM')
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # Create file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def set_seed(seed=42):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch (int): Current epoch
        metrics (dict): Metrics to save
        checkpoint_path (str): Path to save checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model, checkpoint_path, optimizer=None, device='cuda'):
    """
    Load model checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_path (str): Path to checkpoint
        optimizer: Optimizer to load state into (optional)
        device (str): Device to load model on

    Returns:
        dict: Checkpoint data
    """
    # PyTorch 2.6+ defaults to weights_only=True, but checkpoints may contain numpy objects
    # Since these are our own checkpoints, we can safely set weights_only=False
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions that don't have weights_only parameter
        checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded from {checkpoint_path}")

    return checkpoint


def save_config(config, save_path):
    """
    Save configuration to JSON file.

    Args:
        config (dict): Configuration dictionary
        save_path (str): Path to save config
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Config saved to {save_path}")


def load_config(config_path):
    """
    Load configuration from JSON file.

    Args:
        config_path (str): Path to config file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricTracker:
    """
    Track multiple metrics during training.
    """

    def __init__(self, *keys):
        self.metrics = {key: AverageMeter() for key in keys}

    def update(self, key, value, n=1):
        if key in self.metrics:
            self.metrics[key].update(value, n)

    def reset(self):
        for meter in self.metrics.values():
            meter.reset()

    def get_metrics(self):
        return {key: meter.avg for key, meter in self.metrics.items()}

    def __str__(self):
        metrics_str = []
        for key, meter in self.metrics.items():
            metrics_str.append(f"{key}: {meter.avg:.4f}")
        return ", ".join(metrics_str)


def count_parameters(model):
    """
    Count number of trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    """
    Get current learning rate from optimizer.

    Args:
        optimizer: PyTorch optimizer

    Returns:
        float: Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def format_time(seconds):
    """
    Format seconds into readable time string.

    Args:
        seconds (float): Time in seconds

    Returns:
        str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def create_output_dir(base_dir, experiment_name):
    """
    Create output directory for experiment.

    Args:
        base_dir (str): Base directory
        experiment_name (str): Name of experiment

    Returns:
        Path: Path to output directory
    """
    output_dir = Path(base_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    (output_dir / 'results').mkdir(exist_ok=True)

    return output_dir
