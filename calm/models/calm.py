"""
CALM: Class-anchor-ALigned generative Modeling

Main model architecture integrating all components for multi-modal representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .class_anchors import ClassAnchorExtractor
from .probability_distribution import (
    ProbabilityDistributionComputer,
    MultiModalFeatureExtractor
)
from .cross_modal_vae import CrossModalVAE


class CALM(nn.Module):
    """
    CALM: Class-anchor-ALigned generative Modeling

    A novel approach for multi-modal representation learning that:
    1. Uses class anchors from an independent dataset as semantic prompts
    2. Computes probability distributions between modalities and class anchors
    3. Employs a cross-modal VAE to align distributions and model uncertainty

    Args:
        clip_model: Pre-trained CLIP model
        tokenizer: CLIP tokenizer
        class_labels (list): List of class label strings
        num_frames (int): Number of frames to sample from video
        latent_dim (int): Dimension of VAE latent space
        temperature (float): Temperature for probability distribution
        fusion_method (str): Temporal fusion method ('mean', 'max', 'attention')
        vae_hidden_dims (list): Hidden dimensions for VAE
        dropout (float): Dropout rate
        device (str): Device to use
    """

    def __init__(
        self,
        clip_model,
        tokenizer,
        class_labels,
        num_frames=12,
        latent_dim=256,
        temperature=0.07,
        fusion_method='mean',
        vae_hidden_dims=None,
        dropout=0.1,
        device='cuda'
    ):
        super().__init__()

        self.device = device
        self.num_frames = num_frames
        self.num_classes = len(class_labels)

        # Feature extraction module (CLIP-based)
        self.feature_extractor = MultiModalFeatureExtractor(
            clip_model=clip_model,
            num_frames=num_frames,
            fusion_method=fusion_method
        )

        # Class anchor extraction
        self.class_anchor_extractor = ClassAnchorExtractor(
            class_labels=class_labels,
            text_encoder=clip_model.encode_text,
            embed_dim=clip_model.visual.output_dim
        )

        # Initialize class anchors
        self.class_anchor_extractor.encode_anchors(tokenizer, device=device)

        # Probability distribution computer
        self.prob_dist_computer = ProbabilityDistributionComputer(
            temperature=temperature
        )

        # Cross-modal probabilistic VAE
        self.vae = CrossModalVAE(
            num_classes=self.num_classes,
            latent_dim=latent_dim,
            hidden_dims=vae_hidden_dims,
            dropout=dropout
        )

        # Move to device
        self.to(device)

    def encode_video(self, video_frames):
        """
        Encode video frames to features.

        Args:
            video_frames (torch.Tensor): Video frames [batch_size, num_frames, C, H, W]

        Returns:
            torch.Tensor: Video features [batch_size, embed_dim]
        """
        return self.feature_extractor.encode_video(video_frames)

    def encode_text(self, text_tokens):
        """
        Encode text tokens to features.

        Args:
            text_tokens (torch.Tensor): Tokenized text [batch_size, seq_len]

        Returns:
            torch.Tensor: Text features [batch_size, embed_dim]
        """
        return self.feature_extractor.encode_text(text_tokens)

    def compute_probability_distributions(self, video_features, text_features):
        """
        Compute probability distributions over class anchors.

        Args:
            video_features (torch.Tensor): Video features [batch_size, embed_dim]
            text_features (torch.Tensor): Text features [batch_size, embed_dim]

        Returns:
            tuple: (video_prob_dist, text_prob_dist)
        """
        # Get class anchors
        class_anchors = self.class_anchor_extractor()

        # Compute probability distributions (Equations 6-7)
        video_prob_dist, text_prob_dist = self.prob_dist_computer(
            video_features,
            text_features,
            class_anchors
        )

        return video_prob_dist, text_prob_dist

    def forward(self, video_frames=None, text_tokens=None, return_embeddings=False):
        """
        Forward pass through CALM model.

        Args:
            video_frames (torch.Tensor, optional): Video frames [B, T, C, H, W]
            text_tokens (torch.Tensor, optional): Text tokens [B, seq_len]
            return_embeddings (bool): Whether to return raw embeddings

        Returns:
            dict: Model outputs including:
                - video_features: Video embeddings
                - text_features: Text embeddings
                - video_prob_dist: Video-anchor probability distribution
                - text_prob_dist: Text-anchor probability distribution
                - recon_text_prob: Reconstructed text probability from VAE
                - mu: VAE latent mean
                - logvar: VAE latent log variance
                - z: VAE latent variable
        """
        outputs = {}

        # Extract features
        if video_frames is not None:
            video_features = self.encode_video(video_frames)
            outputs['video_features'] = video_features
        else:
            video_features = None

        if text_tokens is not None:
            text_features = self.encode_text(text_tokens)
            outputs['text_features'] = text_features
        else:
            text_features = None

        # Compute probability distributions
        if video_features is not None and text_features is not None:
            video_prob_dist, text_prob_dist = self.compute_probability_distributions(
                video_features, text_features
            )

            outputs['video_prob_dist'] = video_prob_dist
            outputs['text_prob_dist'] = text_prob_dist

            # Cross-modal VAE: reconstruct text-anchor distribution from video-anchor distribution
            vae_outputs = self.vae(video_prob_dist)

            outputs.update({
                'recon_text_prob': vae_outputs['recon_text_prob'],
                'mu': vae_outputs['mu'],
                'logvar': vae_outputs['logvar'],
                'z': vae_outputs['z']
            })

        if return_embeddings:
            return outputs

        return outputs

    def get_video_embeddings(self, video_frames):
        """
        Get video embeddings for retrieval.

        Args:
            video_frames (torch.Tensor): Video frames

        Returns:
            torch.Tensor: Normalized video embeddings
        """
        with torch.no_grad():
            video_features = self.encode_video(video_frames)
            video_features = F.normalize(video_features, p=2, dim=-1)
        return video_features

    def get_text_embeddings(self, text_tokens):
        """
        Get text embeddings for retrieval.

        Args:
            text_tokens (torch.Tensor): Text tokens

        Returns:
            torch.Tensor: Normalized text embeddings
        """
        with torch.no_grad():
            text_features = self.encode_text(text_tokens)
            text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features

    def compute_similarity(self, video_frames, text_tokens):
        """
        Compute similarity between video and text.

        Args:
            video_frames (torch.Tensor): Video frames
            text_tokens (torch.Tensor): Text tokens

        Returns:
            torch.Tensor: Similarity scores
        """
        video_features = self.get_video_embeddings(video_frames)
        text_features = self.get_text_embeddings(text_tokens)

        similarity = torch.matmul(video_features, text_features.t())
        return similarity

    @torch.no_grad()
    def retrieve(self, video_frames, text_tokens, top_k=5):
        """
        Retrieve top-k videos for given text queries (or vice versa).

        Args:
            video_frames (torch.Tensor): Video frames [num_videos, T, C, H, W]
            text_tokens (torch.Tensor): Text tokens [num_texts, seq_len]
            top_k (int): Number of top results to return

        Returns:
            dict: Retrieval results
        """
        # Compute similarity matrix
        similarity = self.compute_similarity(video_frames, text_tokens)

        # Text-to-video retrieval
        t2v_scores, t2v_indices = torch.topk(similarity, k=top_k, dim=0)

        # Video-to-text retrieval
        v2t_scores, v2t_indices = torch.topk(similarity, k=top_k, dim=1)

        return {
            't2v_indices': t2v_indices,  # [top_k, num_texts]
            't2v_scores': t2v_scores,
            'v2t_indices': v2t_indices,  # [num_videos, top_k]
            'v2t_scores': v2t_scores,
            'similarity': similarity
        }


class CALMForRetrieval(CALM):
    """
    CALM model specifically configured for video-text retrieval tasks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, video_frames, text_tokens):
        """
        Forward pass for retrieval task.

        Args:
            video_frames (torch.Tensor): Video frames
            text_tokens (torch.Tensor): Text tokens

        Returns:
            dict: Outputs for retrieval
        """
        return super().forward(
            video_frames=video_frames,
            text_tokens=text_tokens,
            return_embeddings=True
        )


class CALMForCaptioning(nn.Module):
    """
    CALM model extended for video captioning tasks.

    Adds a transformer decoder on top of CALM for caption generation.

    Args:
        calm_model: Pre-trained CALM model
        vocab_size (int): Vocabulary size
        max_seq_len (int): Maximum sequence length
        num_decoder_layers (int): Number of decoder layers
        num_heads (int): Number of attention heads
    """

    def __init__(
        self,
        calm_model,
        vocab_size,
        max_seq_len=30,
        num_decoder_layers=6,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.calm = calm_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        embed_dim = calm_model.feature_extractor.embed_dim

        # Caption decoder
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        self.output_projection = nn.Linear(embed_dim, vocab_size)

    def forward(self, video_frames, caption_tokens=None):
        """
        Forward pass for captioning.

        Args:
            video_frames (torch.Tensor): Video frames
            caption_tokens (torch.Tensor, optional): Caption tokens for training

        Returns:
            dict: Outputs including caption logits
        """
        # Get video features from CALM
        outputs = self.calm(video_frames=video_frames, return_embeddings=True)
        video_features = outputs['video_features']  # [B, D]

        # Expand video features as memory for decoder
        video_memory = video_features.unsqueeze(1)  # [B, 1, D]

        if caption_tokens is not None:
            # Training mode
            batch_size, seq_len = caption_tokens.shape

            # Embed caption tokens
            token_embeds = self.token_embedding(caption_tokens)
            token_embeds = token_embeds + self.pos_embedding[:, :seq_len, :]

            # Create causal mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(
                video_frames.device
            )

            # Decode
            decoder_out = self.decoder(
                tgt=token_embeds,
                memory=video_memory,
                tgt_mask=tgt_mask
            )

            # Project to vocabulary
            logits = self.output_projection(decoder_out)

            outputs['caption_logits'] = logits

        return outputs

    @torch.no_grad()
    def generate(self, video_frames, start_token_id, end_token_id, max_len=None):
        """
        Generate captions for videos using greedy decoding.

        Args:
            video_frames (torch.Tensor): Video frames
            start_token_id (int): Start token ID
            end_token_id (int): End token ID
            max_len (int, optional): Maximum generation length

        Returns:
            torch.Tensor: Generated caption token IDs
        """
        if max_len is None:
            max_len = self.max_seq_len

        batch_size = video_frames.size(0)
        device = video_frames.device

        # Get video features
        outputs = self.calm(video_frames=video_frames, return_embeddings=True)
        video_features = outputs['video_features']
        video_memory = video_features.unsqueeze(1)

        # Initialize with start token
        generated = torch.full(
            (batch_size, 1),
            start_token_id,
            dtype=torch.long,
            device=device
        )

        for _ in range(max_len - 1):
            # Embed current tokens
            token_embeds = self.token_embedding(generated)
            seq_len = token_embeds.size(1)
            token_embeds = token_embeds + self.pos_embedding[:, :seq_len, :]

            # Decode
            decoder_out = self.decoder(
                tgt=token_embeds,
                memory=video_memory
            )

            # Get next token
            logits = self.output_projection(decoder_out[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check if all sequences have ended
            if (next_token == end_token_id).all():
                break

        return generated
