"""
Probability Distribution Module for CALM
Computes class probability distributions for video and text features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbabilityDistributionComputer(nn.Module):
    """
    Computes probability distributions over class anchors for each modality.

    This module calculates cosine similarity between modality features and class anchors,
    then applies softmax to obtain probability distributions (Equations 6-7 in paper).

    Args:
        temperature (float): Temperature parameter for softmax sharpness
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def compute_cosine_similarity(self, features, anchors):
        """
        Compute cosine similarity between features and anchors.

        Args:
            features (torch.Tensor): Input features [batch_size, embed_dim]
            anchors (torch.Tensor): Class anchors [num_classes, embed_dim]

        Returns:
            torch.Tensor: Similarity scores [batch_size, num_classes]
        """
        # Normalize features and anchors
        features_norm = F.normalize(features, p=2, dim=-1)
        anchors_norm = F.normalize(anchors, p=2, dim=-1)

        # Compute cosine similarity
        similarity = torch.matmul(features_norm, anchors_norm.t())

        return similarity

    def compute_probability_distribution(self, features, anchors):
        """
        Compute probability distribution over class anchors.

        Args:
            features (torch.Tensor): Input features [batch_size, embed_dim]
            anchors (torch.Tensor): Class anchors [num_classes, embed_dim]

        Returns:
            torch.Tensor: Probability distribution [batch_size, num_classes]
        """
        # Compute cosine similarity
        similarity = self.compute_cosine_similarity(features, anchors)

        # Apply temperature scaling and softmax (Equation 6)
        probability_dist = F.softmax(similarity / self.temperature, dim=-1)

        return probability_dist

    def forward(self, video_features, text_features, class_anchors):
        """
        Compute probability distributions for both video and text modalities.

        Args:
            video_features (torch.Tensor): Video features [batch_size, embed_dim]
            text_features (torch.Tensor): Text features [batch_size, embed_dim]
            class_anchors (torch.Tensor): Class anchors [num_classes, embed_dim]

        Returns:
            tuple: (video_prob_dist, text_prob_dist)
                - video_prob_dist: [batch_size, num_classes] (inter-modal, Vp)
                - text_prob_dist: [batch_size, num_classes] (intra-modal, Sp)
        """
        # Compute video-anchor probability distribution (inter-modal)
        video_prob_dist = self.compute_probability_distribution(video_features, class_anchors)

        # Compute text-anchor probability distribution (intra-modal)
        text_prob_dist = self.compute_probability_distribution(text_features, class_anchors)

        return video_prob_dist, text_prob_dist


class TemporalFeatureFusion(nn.Module):
    """
    Temporal fusion module to aggregate frame-level features into video-level features.

    Implements Equation 2 in the paper: V = ΨTE(h_v_1, h_v_2, ..., h_v_T)

    Args:
        embed_dim (int): Feature embedding dimension
        fusion_method (str): Method for temporal fusion ('mean', 'max', 'attention')
    """

    def __init__(self, embed_dim=512, fusion_method='mean'):
        super().__init__()
        self.fusion_method = fusion_method
        self.embed_dim = embed_dim

        if fusion_method == 'attention':
            # Temporal attention mechanism
            self.attention_weights = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, 1)
            )

    def forward(self, frame_features):
        """
        Fuse frame-level features into video-level representation.

        Args:
            frame_features (torch.Tensor): Frame features [batch_size, num_frames, embed_dim]

        Returns:
            torch.Tensor: Video-level features [batch_size, embed_dim]
        """
        if self.fusion_method == 'mean':
            # Simple mean pooling across temporal dimension
            video_features = torch.mean(frame_features, dim=1)

        elif self.fusion_method == 'max':
            # Max pooling across temporal dimension
            video_features = torch.max(frame_features, dim=1)[0]

        elif self.fusion_method == 'attention':
            # Attention-based temporal fusion
            # Compute attention weights for each frame
            attn_scores = self.attention_weights(frame_features)  # [B, T, 1]
            attn_weights = F.softmax(attn_scores, dim=1)  # [B, T, 1]

            # Weighted sum of frame features
            video_features = torch.sum(frame_features * attn_weights, dim=1)  # [B, D]

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        return video_features


class MultiModalFeatureExtractor(nn.Module):
    """
    Feature extraction module for video and text modalities using CLIP.

    Args:
        clip_model: Pre-trained CLIP model
        num_frames (int): Number of frames to sample from video
        fusion_method (str): Temporal fusion method
    """

    def __init__(self, clip_model, num_frames=12, fusion_method='mean'):
        super().__init__()
        self.clip_model = clip_model
        self.num_frames = num_frames

        # Get embedding dimension from CLIP
        self.embed_dim = clip_model.visual.output_dim

        # Temporal fusion module
        self.temporal_fusion = TemporalFeatureFusion(
            embed_dim=self.embed_dim,
            fusion_method=fusion_method
        )

    def encode_video(self, video_frames):
        """
        Encode video frames into video-level features.

        Args:
            video_frames (torch.Tensor): Video frames [batch_size, num_frames, C, H, W]

        Returns:
            torch.Tensor: Video features [batch_size, embed_dim]
        """
        batch_size, num_frames = video_frames.shape[:2]

        # Reshape to process all frames at once
        frames = video_frames.view(-1, *video_frames.shape[2:])  # [B*T, C, H, W]

        # Encode frames using CLIP visual encoder (Equation 1)
        with torch.no_grad():
            frame_features = self.clip_model.encode_image(frames)  # [B*T, D]

        # Reshape back to batch and temporal dimensions
        frame_features = frame_features.view(batch_size, num_frames, -1)  # [B, T, D]

        # Aggregate frame features using temporal fusion (Equation 2)
        video_features = self.temporal_fusion(frame_features)  # [B, D]

        return video_features

    def encode_text(self, text_tokens):
        """
        Encode text tokens into text features.

        Args:
            text_tokens (torch.Tensor): Tokenized text [batch_size, seq_len]

        Returns:
            torch.Tensor: Text features [batch_size, embed_dim]
        """
        # Encode text using CLIP text encoder (Equation 3)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)  # [B, D]

        return text_features

    def forward(self, video_frames=None, text_tokens=None):
        """
        Extract features from video and/or text.

        Args:
            video_frames (torch.Tensor, optional): Video frames
            text_tokens (torch.Tensor, optional): Text tokens

        Returns:
            dict: Dictionary containing extracted features
        """
        outputs = {}

        if video_frames is not None:
            outputs['video_features'] = self.encode_video(video_frames)

        if text_tokens is not None:
            outputs['text_features'] = self.encode_text(text_tokens)

        return outputs
