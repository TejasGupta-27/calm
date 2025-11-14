"""
Evaluation Script for CALM

Evaluate trained CALM model on video-text retrieval tasks.
"""

import os
import sys
import argparse
from pathlib import Path
import json

import torch
import clip

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calm.models import CALMForRetrieval
from calm.models.class_anchors import load_class_labels
from calm.data import create_dataloader
from calm.utils import (
    setup_logging,
    set_seed,
    load_checkpoint,
    RetrievalEvaluator
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate CALM model')

    # Dataset
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['msrvtt', 'didemo', 'msvd', 'lsmdc'],
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing videos')
    parser.add_argument('--annotation_file', type=str, required=True,
                        help='Path to test annotations')

    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                        help='CLIP model variant')
    parser.add_argument('--num_frames', type=int, default=12,
                        help='Number of frames to sample')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='VAE latent dimension')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for probability distributions')

    # Evaluation
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (reduce if OOM)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers (reduce if OOM)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (None for all)')

    # Output
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save results')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()
    return args


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate model on test set.

    Args:
        model: CALM model
        dataloader: Test dataloader
        device: Device

    Returns:
        dict: Evaluation metrics
    """
    model.eval()

    evaluator = RetrievalEvaluator(top_k=[1, 5, 10])

    print("Running evaluation...")
    for batch_idx, batch in enumerate(dataloader):
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

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
    print("\nComputing metrics...")
    metrics = evaluator.compute()

    return metrics


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load CLIP model
    print(f"Loading CLIP model: {args.clip_model}")
    clip_model, preprocess = clip.load(args.clip_model, device=device)

    # Load class labels
    print("Loading class labels")
    class_labels = load_class_labels('charades')

    # Create CALM model
    print("Creating CALM model")
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

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    load_checkpoint(model, args.checkpoint, device=device)

    # Create dataloader
    print("Creating dataloader")
    test_loader = create_dataloader(
        dataset_name=args.dataset,
        data_root=args.data_root,
        annotation_file=args.annotation_file,
        num_frames=args.num_frames,
        frame_size=(224, 224),
        split='test',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=preprocess,
        max_samples=args.max_samples
    )

    print(f"Test samples: {len(test_loader.dataset)}")

    # Evaluate
    metrics = evaluate(model, test_loader, device)

    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)

    print("\nText-to-Video Retrieval:")
    print(f"  R@1:  {metrics['t2v_r1']:.2f}")
    print(f"  R@5:  {metrics['t2v_r5']:.2f}")
    print(f"  R@10: {metrics['t2v_r10']:.2f}")
    print(f"  Mean Rank: {metrics['t2v_mean_rank']:.2f}")

    print("\nVideo-to-Text Retrieval:")
    print(f"  R@1:  {metrics['v2t_r1']:.2f}")
    print(f"  R@5:  {metrics['v2t_r5']:.2f}")
    print(f"  R@10: {metrics['v2t_r10']:.2f}")
    print(f"  Mean Rank: {metrics['v2t_mean_rank']:.2f}")

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
