"""
Loss Functions for CALM

Implements the training objectives including reconstruction loss, KL divergence,
and task-specific losses for video retrieval and captioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cross_modal_vae import compute_kl_divergence, compute_reconstruction_loss


class CALMLoss(nn.Module):
    """
    Combined loss function for CALM framework.

    Implements Equation 16: L = L_rec + α * L_KL + L_task

    Args:
        alpha (float): Weight for KL divergence loss
        task (str): Task type ('retrieval' or 'captioning')
        temperature (float): Temperature for contrastive loss
    """

    def __init__(self, alpha=0.1, task='retrieval', temperature=0.07):
        super().__init__()
        self.alpha = alpha
        self.task = task
        self.temperature = temperature

    def forward(self, outputs, batch):
        """
        Compute total loss.

        Args:
            outputs (dict): Model outputs containing:
                - recon_text_prob: Reconstructed text probability
                - mu: VAE latent mean
                - logvar: VAE latent log variance
                - video_features: Video embeddings (for retrieval)
                - text_features: Text embeddings (for retrieval)
                - captions: Generated captions (for captioning)
            batch (dict): Batch data

        Returns:
            dict: Dictionary of losses
        """
        # Reconstruction loss (Equation 12)
        recon_loss = compute_reconstruction_loss(
            outputs['recon_text_prob'],
            outputs['text_prob_dist']
        ).mean()

        # KL divergence loss (Equation 15)
        kl_loss = compute_kl_divergence(
            outputs['mu'],
            outputs['logvar']
        ).mean()

        # Task-specific loss
        if self.task == 'retrieval':
            task_loss = self.compute_retrieval_loss(outputs, batch)
        elif self.task == 'captioning':
            task_loss = self.compute_captioning_loss(outputs, batch)
        else:
            task_loss = torch.tensor(0.0, device=recon_loss.device)

        # Total loss (Equation 16)
        total_loss = recon_loss + self.alpha * kl_loss + task_loss

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'task_loss': task_loss
        }

    def compute_retrieval_loss(self, outputs, batch):
        """
        Compute contrastive loss for video-text retrieval.

        Uses InfoNCE loss to align video and text embeddings.

        Args:
            outputs (dict): Model outputs
            batch (dict): Batch data

        Returns:
            torch.Tensor: Retrieval loss
        """
        video_features = outputs['video_features']
        text_features = outputs['text_features']

        # Normalize features
        video_features = F.normalize(video_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # Compute similarity matrix
        similarity = torch.matmul(video_features, text_features.t()) / self.temperature

        # Labels: diagonal elements are positive pairs
        batch_size = video_features.size(0)
        labels = torch.arange(batch_size, device=video_features.device)

        # Symmetric loss (video-to-text + text-to-video)
        loss_v2t = F.cross_entropy(similarity, labels)
        loss_t2v = F.cross_entropy(similarity.t(), labels)

        retrieval_loss = (loss_v2t + loss_t2v) / 2

        return retrieval_loss

    def compute_captioning_loss(self, outputs, batch):
        """
        Compute cross-entropy loss for video captioning.

        Args:
            outputs (dict): Model outputs containing logits
            batch (dict): Batch data containing target captions

        Returns:
            torch.Tensor: Captioning loss
        """
        if 'caption_logits' not in outputs or 'caption_targets' not in batch:
            return torch.tensor(0.0, device=outputs['mu'].device)

        logits = outputs['caption_logits']  # [B, seq_len, vocab_size]
        targets = batch['caption_targets']  # [B, seq_len]

        # Reshape for cross-entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        targets = targets.view(-1)

        # Compute cross-entropy loss
        caption_loss = F.cross_entropy(
            logits,
            targets,
            ignore_index=0  # Ignore padding tokens
        )

        return caption_loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for video-text alignment (InfoNCE).

    Args:
        temperature (float): Temperature parameter
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, video_features, text_features):
        """
        Compute bidirectional contrastive loss.

        Args:
            video_features (torch.Tensor): Video embeddings [batch_size, dim]
            text_features (torch.Tensor): Text embeddings [batch_size, dim]

        Returns:
            torch.Tensor: Contrastive loss
        """
        # Normalize features
        video_features = F.normalize(video_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # Compute similarity matrix
        similarity = torch.matmul(video_features, text_features.t()) / self.temperature

        # Create labels (diagonal elements are positive pairs)
        batch_size = video_features.size(0)
        labels = torch.arange(batch_size, device=video_features.device)

        # Video-to-text and text-to-video losses
        loss_v2t = F.cross_entropy(similarity, labels)
        loss_t2v = F.cross_entropy(similarity.t(), labels)

        # Average the two directions
        loss = (loss_v2t + loss_t2v) / 2

        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for multi-modal alignment.

    Args:
        margin (float): Margin for triplet loss
    """

    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss.

        Args:
            anchor (torch.Tensor): Anchor embeddings
            positive (torch.Tensor): Positive embeddings
            negative (torch.Tensor): Negative embeddings

        Returns:
            torch.Tensor: Triplet loss
        """
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        # Triplet loss with margin
        loss = F.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()


def compute_retrieval_metrics(video_features, text_features, top_k=[1, 5, 10]):
    """
    Compute retrieval metrics (Recall@K).

    Args:
        video_features (torch.Tensor): Video embeddings [num_videos, dim]
        text_features (torch.Tensor): Text embeddings [num_texts, dim]
        top_k (list): List of K values for Recall@K

    Returns:
        dict: Dictionary of metrics
    """
    # Normalize features
    video_features = F.normalize(video_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)

    # Compute similarity matrix
    similarity = torch.matmul(video_features, text_features.t())

    # Text-to-video retrieval
    t2v_ranks = []
    for i in range(similarity.size(1)):
        sims = similarity[:, i]
        sorted_indices = torch.argsort(sims, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        t2v_ranks.append(rank)

    # Video-to-text retrieval
    v2t_ranks = []
    for i in range(similarity.size(0)):
        sims = similarity[i, :]
        sorted_indices = torch.argsort(sims, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        v2t_ranks.append(rank)

    t2v_ranks = torch.tensor(t2v_ranks)
    v2t_ranks = torch.tensor(v2t_ranks)

    # Compute metrics
    metrics = {}
    for k in top_k:
        metrics[f't2v_r{k}'] = (t2v_ranks < k).float().mean().item() * 100
        metrics[f'v2t_r{k}'] = (v2t_ranks < k).float().mean().item() * 100

    # Mean rank
    metrics['t2v_mean_rank'] = t2v_ranks.float().mean().item()
    metrics['v2t_mean_rank'] = v2t_ranks.float().mean().item()

    # Median rank
    metrics['t2v_median_rank'] = t2v_ranks.float().median().item()
    metrics['v2t_median_rank'] = v2t_ranks.float().median().item()

    return metrics
