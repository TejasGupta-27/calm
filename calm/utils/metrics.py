"""
Evaluation Metrics for CALM

Implements metrics for video retrieval and captioning tasks.
"""

import numpy as np
import torch
import torch.nn.functional as F


def compute_retrieval_metrics(video_embeddings, text_embeddings, top_k=[1, 5, 10]):
    """
    Compute retrieval metrics for video-text retrieval.

    Args:
        video_embeddings (torch.Tensor): Video embeddings [num_videos, dim]
        text_embeddings (torch.Tensor): Text embeddings [num_texts, dim]
        top_k (list): List of K values for Recall@K

    Returns:
        dict: Dictionary containing retrieval metrics
    """
    # Normalize embeddings
    video_embeddings = F.normalize(video_embeddings, p=2, dim=-1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

    # Compute similarity matrix
    similarity = torch.matmul(video_embeddings, text_embeddings.t())  # [V, T]

    # Text-to-video retrieval
    t2v_metrics = compute_directional_metrics(similarity.t(), top_k)

    # Video-to-text retrieval
    v2t_metrics = compute_directional_metrics(similarity, top_k)

    # Combine metrics
    metrics = {
        **{f't2v_{k}': v for k, v in t2v_metrics.items()},
        **{f'v2t_{k}': v for k, v in v2t_metrics.items()}
    }

    return metrics


def compute_directional_metrics(similarity, top_k=[1, 5, 10]):
    """
    Compute retrieval metrics in one direction.

    Args:
        similarity (torch.Tensor): Similarity matrix [num_queries, num_targets]
        top_k (list): List of K values

    Returns:
        dict: Metrics dictionary
    """
    num_queries = similarity.size(0)
    ranks = []

    for i in range(num_queries):
        # Get similarity scores for this query
        sims = similarity[i]

        # Sort in descending order
        sorted_indices = torch.argsort(sims, descending=True)

        # Find rank of correct target (assuming diagonal correspondence)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        ranks.append(rank)

    ranks = np.array(ranks)

    # Compute recall at K
    metrics = {}
    for k in top_k:
        metrics[f'r{k}'] = (ranks < k).mean() * 100

    # Mean rank
    metrics['mean_rank'] = ranks.mean()

    # Median rank
    metrics['median_rank'] = np.median(ranks)

    return metrics


def compute_captioning_metrics(generated_captions, reference_captions):
    """
    Compute captioning metrics (BLEU, METEOR, ROUGE-L, CIDEr).

    Note: This requires external libraries like pycocoevalcap.
    For simplicity, this is a placeholder that should be replaced with actual implementations.

    Args:
        generated_captions (list): List of generated caption strings
        reference_captions (list): List of reference caption strings or lists

    Returns:
        dict: Dictionary of captioning metrics
    """
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider

        # Format data for evaluation
        gts = {}  # Ground truth captions
        res = {}  # Generated captions

        for idx, (gen_cap, ref_caps) in enumerate(zip(generated_captions, reference_captions)):
            res[idx] = [gen_cap]
            if isinstance(ref_caps, list):
                gts[idx] = ref_caps
            else:
                gts[idx] = [ref_caps]

        # Compute metrics
        scorers = [
            (Bleu(4), "BLEU"),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        metrics = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, res)
            if isinstance(score, list):
                # BLEU returns multiple scores
                metrics[f'{method}_1'] = score[0]
                metrics[f'{method}_2'] = score[1]
                metrics[f'{method}_3'] = score[2]
                metrics[f'{method}_4'] = score[3]
            else:
                metrics[method] = score

        return metrics

    except ImportError:
        print("Warning: pycocoevalcap not installed. Returning dummy metrics.")
        return {
            'BLEU_4': 0.0,
            'METEOR': 0.0,
            'ROUGE_L': 0.0,
            'CIDEr': 0.0
        }


def accuracy_at_k(predictions, targets, k=5):
    """
    Compute top-k accuracy.

    Args:
        predictions (torch.Tensor): Predicted scores [batch_size, num_classes]
        targets (torch.Tensor): Ground truth labels [batch_size]
        k (int): Top K

    Returns:
        float: Top-k accuracy
    """
    batch_size = targets.size(0)
    _, top_k_pred = predictions.topk(k, dim=1, largest=True, sorted=True)

    correct = top_k_pred.eq(targets.view(-1, 1).expand_as(top_k_pred))
    correct_k = correct.view(-1).float().sum(0)

    return correct_k.item() / batch_size


def mean_reciprocal_rank(similarity, targets=None):
    """
    Compute Mean Reciprocal Rank (MRR).

    Args:
        similarity (torch.Tensor): Similarity matrix [num_queries, num_targets]
        targets (torch.Tensor, optional): Target indices for each query

    Returns:
        float: MRR score
    """
    if targets is None:
        # Assume diagonal correspondence
        targets = torch.arange(similarity.size(0))

    reciprocal_ranks = []

    for i, target in enumerate(targets):
        # Get similarity scores for this query
        sims = similarity[i]

        # Sort in descending order
        sorted_indices = torch.argsort(sims, descending=True)

        # Find rank of correct target (1-indexed)
        rank = (sorted_indices == target).nonzero(as_tuple=True)[0].item() + 1
        reciprocal_ranks.append(1.0 / rank)

    return np.mean(reciprocal_ranks)


class RetrievalEvaluator:
    """
    Evaluator for video-text retrieval tasks.
    """

    def __init__(self, top_k=[1, 5, 10]):
        self.top_k = top_k
        self.reset()

    def reset(self):
        self.video_embeddings = []
        self.text_embeddings = []

    def add_batch(self, video_emb, text_emb):
        """
        Add batch of embeddings.

        Args:
            video_emb (torch.Tensor): Video embeddings
            text_emb (torch.Tensor): Text embeddings
        """
        self.video_embeddings.append(video_emb.cpu())
        self.text_embeddings.append(text_emb.cpu())

    def compute(self):
        """
        Compute retrieval metrics.

        Returns:
            dict: Retrieval metrics
        """
        # Concatenate all embeddings
        video_embeddings = torch.cat(self.video_embeddings, dim=0)
        text_embeddings = torch.cat(self.text_embeddings, dim=0)

        # Compute metrics
        metrics = compute_retrieval_metrics(
            video_embeddings,
            text_embeddings,
            top_k=self.top_k
        )

        return metrics


class CaptioningEvaluator:
    """
    Evaluator for video captioning tasks.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.generated_captions = []
        self.reference_captions = []

    def add_batch(self, generated, references):
        """
        Add batch of captions.

        Args:
            generated (list): Generated captions
            references (list): Reference captions
        """
        self.generated_captions.extend(generated)
        self.reference_captions.extend(references)

    def compute(self):
        """
        Compute captioning metrics.

        Returns:
            dict: Captioning metrics
        """
        metrics = compute_captioning_metrics(
            self.generated_captions,
            self.reference_captions
        )

        return metrics
