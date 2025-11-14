"""
Demo Script for CALM

A simple demonstration of how to use CALM for video-text retrieval.
"""

import sys
from pathlib import Path
import torch
import clip
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calm.models import CALMForRetrieval
from calm.models.class_anchors import load_class_labels


def create_dummy_video(num_frames=12, size=224):
    """
    Create dummy video frames for demonstration.

    Returns:
        torch.Tensor: Random video frames [1, num_frames, 3, size, size]
    """
    video = torch.randn(1, num_frames, 3, size, size)
    return video


def main():
    """Main demo function."""
    print("="*60)
    print("CALM Demo: Video-Text Retrieval")
    print("="*60)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n1. Using device: {device}")

    # Load CLIP
    print("\n2. Loading CLIP model (ViT-B/32)...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    print("   ✓ CLIP loaded successfully")

    # Load class labels
    print("\n3. Loading class anchors from Charades dataset...")
    class_labels = load_class_labels('charades')
    print(f"   ✓ Loaded {len(class_labels)} class anchors")

    # Create CALM model
    print("\n4. Creating CALM model...")
    model = CALMForRetrieval(
        clip_model=clip_model,
        tokenizer=clip.tokenize,
        class_labels=class_labels,
        num_frames=12,
        latent_dim=256,
        temperature=0.07,
        fusion_method='mean',
        dropout=0.1,
        device=device
    )
    print("   ✓ CALM model created")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   → Total trainable parameters: {num_params:,}")

    # Example texts
    texts = [
        "a person is walking down the street",
        "a cat is playing with a ball",
        "someone is cooking in the kitchen",
        "a man is talking on the phone"
    ]

    print("\n5. Encoding texts...")
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_embeddings = model.get_text_embeddings(text_tokens)
    print(f"   ✓ Text embeddings: {text_embeddings.shape}")

    # Create dummy videos
    print("\n6. Creating dummy video data...")
    num_videos = 4
    video_frames = create_dummy_video(num_frames=12).repeat(num_videos, 1, 1, 1, 1).to(device)
    print(f"   ✓ Video frames: {video_frames.shape}")

    # Encode videos
    print("\n7. Encoding videos...")
    with torch.no_grad():
        video_embeddings = model.get_video_embeddings(video_frames)
    print(f"   ✓ Video embeddings: {video_embeddings.shape}")

    # Compute similarity
    print("\n8. Computing video-text similarity...")
    similarity = torch.matmul(video_embeddings, text_embeddings.t())
    print(f"   ✓ Similarity matrix: {similarity.shape}")

    # Show results
    print("\n9. Similarity scores:")
    print("-" * 60)
    similarity_np = similarity.cpu().numpy()

    for i, text in enumerate(texts):
        print(f"\nText {i}: '{text}'")
        for j in range(num_videos):
            score = similarity_np[j, i]
            print(f"  Video {j}: {score:.4f}")

    # Text-to-Video retrieval example
    print("\n10. Text-to-Video Retrieval Example:")
    print("-" * 60)
    query_idx = 0
    query_text = texts[query_idx]
    print(f"Query: '{query_text}'")

    # Get similarity scores for this query
    scores = similarity[:, query_idx].cpu().numpy()
    ranked_indices = np.argsort(scores)[::-1]  # Sort in descending order

    print("\nTop-3 Retrieved Videos:")
    for rank, video_idx in enumerate(ranked_indices[:3], 1):
        print(f"  {rank}. Video {video_idx} (score: {scores[video_idx]:.4f})")

    # Model components demonstration
    print("\n11. Model Components:")
    print("-" * 60)

    # Get class anchors
    class_anchors = model.class_anchor_extractor()
    print(f"   Class anchors shape: {class_anchors.shape}")

    # Compute probability distributions
    with torch.no_grad():
        video_prob, text_prob = model.compute_probability_distributions(
            video_embeddings, text_embeddings
        )
    print(f"   Video probability distribution: {video_prob.shape}")
    print(f"   Text probability distribution: {text_prob.shape}")

    # VAE reconstruction
    with torch.no_grad():
        vae_outputs = model.vae(video_prob)

    print(f"   VAE latent mean: {vae_outputs['mu'].shape}")
    print(f"   VAE latent logvar: {vae_outputs['logvar'].shape}")
    print(f"   Reconstructed text probability: {vae_outputs['recon_text_prob'].shape}")

    # Show top class predictions
    print("\n12. Top Class Anchor Predictions:")
    print("-" * 60)

    # Show for first video
    video_idx = 0
    top_k = 5
    video_probs = video_prob[video_idx].cpu().numpy()
    top_indices = np.argsort(video_probs)[::-1][:top_k]

    print(f"\nVideo {video_idx} - Top {top_k} Class Anchors:")
    for rank, class_idx in enumerate(top_indices, 1):
        class_name = class_labels[class_idx]
        prob = video_probs[class_idx]
        print(f"  {rank}. {class_name}: {prob:.4f}")

    # Show for first text
    text_idx = 0
    text_probs = text_prob[text_idx].cpu().numpy()
    top_indices = np.argsort(text_probs)[::-1][:top_k]

    print(f"\nText {text_idx} ('{texts[text_idx]}') - Top {top_k} Class Anchors:")
    for rank, class_idx in enumerate(top_indices, 1):
        class_name = class_labels[class_idx]
        prob = text_probs[class_idx]
        print(f"  {rank}. {class_name}: {prob:.4f}")

    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Prepare your video-text dataset")
    print("2. Train the model using scripts/train.py")
    print("3. Evaluate using scripts/evaluate.py")
    print("\nSee README.md for detailed instructions.")


if __name__ == '__main__':
    main()
