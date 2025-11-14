"""
Training Script for CALM

Train CALM model for video-text retrieval or captioning tasks.
"""

import os
import sys
import argparse
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calm.models import CALM, CALMForRetrieval, CALMLoss
from calm.models.class_anchors import load_class_labels
from calm.data import create_dataloader
from calm.utils import (
    setup_logging,
    set_seed,
    save_checkpoint,
    save_config,
    MetricTracker,
    count_parameters,
    get_lr,
    format_time,
    create_output_dir,
    RetrievalEvaluator
)
from calm.configs import BaseConfig, MSRVTTConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CALM model')

    # Dataset
    parser.add_argument('--dataset', type=str, default='msrvtt',
                        choices=['msrvtt', 'didemo', 'msvd', 'lsmdc'],
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing videos')
    parser.add_argument('--annotation_train', type=str, required=True,
                        help='Path to training annotations')
    parser.add_argument('--annotation_val', type=str, default=None,
                        help='Path to validation annotations')

    # Model
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                        help='CLIP model variant')
    parser.add_argument('--num_frames', type=int, default=12,
                        help='Number of frames to sample')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='VAE latent dimension')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for probability distributions')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Weight for KL divergence loss')

    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (reduce if OOM: try 8, 16, or 32)')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps (effective batch = batch_size * steps)')

    # Task
    parser.add_argument('--task', type=str, default='retrieval',
                        choices=['retrieval', 'captioning'],
                        help='Task to train for')

    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--experiment_name', type=str, default='calm_experiment',
                        help='Experiment name')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers (reduce if OOM: try 1 or 2)')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Logging interval')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use from dataset (None for all)')

    args = parser.parse_args()
    return args


def train_epoch(model, dataloader, criterion, optimizer, device, logger, epoch, log_interval):
    """
    Train for one epoch.

    Args:
        model: CALM model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device
        logger: Logger
        epoch: Current epoch
        log_interval: Logging interval

    Returns:
        dict: Training metrics
    """
    model.train()

    # Metric tracker
    metrics = MetricTracker('total_loss', 'recon_loss', 'kl_loss', 'task_loss')

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        video_frames = batch['video_frames'].to(device)

        # Tokenize texts
        texts = batch['texts']
        text_tokens = clip.tokenize(texts, truncate=True).to(device)

        # Forward pass
        outputs = model(video_frames=video_frames, text_tokens=text_tokens)

        # Compute loss
        losses = criterion(outputs, batch)

        # Backward pass
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()

        # Update metrics
        batch_size = video_frames.size(0)
        metrics.update('total_loss', losses['total_loss'].item(), batch_size)
        metrics.update('recon_loss', losses['recon_loss'].item(), batch_size)
        metrics.update('kl_loss', losses['kl_loss'].item(), batch_size)
        metrics.update('task_loss', losses['task_loss'].item(), batch_size)

        # Logging
        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            lr = get_lr(optimizer)

            logger.info(
                f"Epoch [{epoch}] Batch [{batch_idx + 1}/{len(dataloader)}] | "
                f"LR: {lr:.2e} | {metrics} | "
                f"Time: {format_time(elapsed)}"
            )

    return metrics.get_metrics()


@torch.no_grad()
def evaluate(model, dataloader, device, logger):
    """
    Evaluate model on validation set.

    Args:
        model: CALM model
        dataloader: Validation dataloader
        device: Device
        logger: Logger

    Returns:
        dict: Evaluation metrics
    """
    model.eval()

    evaluator = RetrievalEvaluator(top_k=[1, 5, 10])

    for batch in dataloader:
        # Move data to device
        video_frames = batch['video_frames'].to(device)
        texts = batch['texts']
        text_tokens = clip.tokenize(texts, truncate=True).to(device)

        # Get embeddings
        video_emb = model.get_video_embeddings(video_frames)
        text_emb = model.get_text_embeddings(text_tokens)

        # Add to evaluator
        evaluator.add_batch(video_emb, text_emb)

    # Compute metrics
    metrics = evaluator.compute()

    # Log metrics
    logger.info("Validation Results:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.2f}")

    return metrics


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output directory
    output_dir = create_output_dir(args.output_dir, args.experiment_name)

    # Setup logging
    logger = setup_logging(output_dir / 'logs', 'train.log')
    logger.info("Starting CALM training")
    logger.info(f"Arguments: {args}")

    # Save config
    save_config(vars(args), output_dir / 'config.json')

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load CLIP model
    logger.info(f"Loading CLIP model: {args.clip_model}")
    clip_model, preprocess = clip.load(args.clip_model, device=device)

    # Load class labels
    logger.info("Loading class labels")
    class_labels = load_class_labels('charades')
    logger.info(f"Number of class anchors: {len(class_labels)}")

    # Create CALM model
    logger.info("Creating CALM model")
    model = CALMForRetrieval(
        clip_model=clip_model,
        tokenizer=clip.tokenize,
        class_labels=class_labels,
        num_frames=args.num_frames,
        latent_dim=args.latent_dim,
        temperature=args.temperature,
        fusion_method='mean',
        dropout=0.1,
        device=device
    )

    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")

    # Create dataloaders
    logger.info("Creating dataloaders")
    train_loader = create_dataloader(
        dataset_name=args.dataset,
        data_root=args.data_root,
        annotation_file=args.annotation_train,
        num_frames=args.num_frames,
        frame_size=(224, 224),
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=preprocess,
        max_samples=args.max_samples
    )

    val_loader = None
    if args.annotation_val:
        # Use proportional max_samples for validation (e.g., 10% of training samples)
        val_max_samples = None
        if args.max_samples is not None:
            val_max_samples = max(100, args.max_samples // 10)  # At least 100 samples

        val_loader = create_dataloader(
            dataset_name=args.dataset,
            data_root=args.data_root,
            annotation_file=args.annotation_val,
            num_frames=args.num_frames,
            frame_size=(224, 224),
            split='val',
            batch_size=32,
            num_workers=args.num_workers,
            transform=preprocess,
            max_samples=val_max_samples
        )

    logger.info(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        logger.info(f"Validation samples: {len(val_loader.dataset)}")

    # Loss function
    criterion = CALMLoss(
        alpha=args.alpha,
        task=args.task,
        temperature=args.temperature
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Training loop
    logger.info("Starting training loop")
    best_metric = 0.0

    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{args.num_epochs}")
        logger.info(f"{'='*50}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            device, logger, epoch, args.log_interval
        )

        # Evaluate
        if val_loader:
            val_metrics = evaluate(model, val_loader, device, logger)

            # Save best model
            if val_metrics['t2v_r1'] > best_metric:
                best_metric = val_metrics['t2v_r1']
                save_checkpoint(
                    model, optimizer, epoch, val_metrics,
                    output_dir / 'checkpoints' / 'best_model.pth'
                )
                logger.info(f"New best model saved! R@1: {best_metric:.2f}")

        # Save checkpoint
        save_checkpoint(
            model, optimizer, epoch, train_metrics,
            output_dir / 'checkpoints' / f'checkpoint_epoch_{epoch}.pth'
        )

    logger.info("\nTraining completed!")
    logger.info(f"Best validation R@1: {best_metric:.2f}")


if __name__ == '__main__':
    main()
