"""
Cross-Modal Probabilistic Variational Autoencoder for CALM

Implements the VAE that reconstructs text-anchor distribution from video-anchor distribution
to model uncertainty and capture deeper relationships between modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder network for VAE.

    Encodes video-anchor probability distribution into latent space.
    Models the approximate posterior q_φ(z|V_p) as a Gaussian distribution.

    Args:
        input_dim (int): Input dimension (number of class anchors)
        latent_dim (int): Latent space dimension
        hidden_dims (list): Hidden layer dimensions
        dropout (float): Dropout rate
    """

    def __init__(self, input_dim, latent_dim=256, hidden_dims=None, dropout=0.1):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 384]

        # Build encoder layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Separate layers for mean and log variance
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, video_prob_dist):
        """
        Encode video-anchor probability distribution.

        Args:
            video_prob_dist (torch.Tensor): Video-anchor probability [batch_size, num_classes]

        Returns:
            tuple: (mu, logvar)
                - mu: Mean of latent distribution [batch_size, latent_dim]
                - logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        h = self.encoder(video_prob_dist)

        # Compute mean and log variance
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder network for VAE.

    Reconstructs text-anchor probability distribution from latent variable z.
    Models p_θ(S_p|z).

    Args:
        latent_dim (int): Latent space dimension
        output_dim (int): Output dimension (number of class anchors)
        hidden_dims (list): Hidden layer dimensions
        dropout (float): Dropout rate
    """

    def __init__(self, latent_dim=256, output_dim=157, hidden_dims=None, dropout=0.1):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [384, 512]

        # Build decoder layers
        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        """
        Decode latent variable to text-anchor probability distribution.

        Args:
            z (torch.Tensor): Latent variable [batch_size, latent_dim]

        Returns:
            torch.Tensor: Reconstructed probability distribution [batch_size, num_classes]
        """
        logits = self.decoder(z)

        # Apply softmax to get probability distribution
        prob_dist = F.softmax(logits, dim=-1)

        return prob_dist


class CrossModalVAE(nn.Module):
    """
    Cross-Modal Probabilistic Variational Autoencoder.

    Reconstructs text-anchor distribution (S_p) from video-anchor distribution (V_p)
    via latent variable z, capturing semantic relationships and uncertainties.

    Implements Section 3.3 of the paper.

    Args:
        num_classes (int): Number of class anchors
        latent_dim (int): Dimension of latent space
        hidden_dims (list): Hidden layer dimensions for encoder/decoder
        dropout (float): Dropout rate
    """

    def __init__(self, num_classes=157, latent_dim=256, hidden_dims=None, dropout=0.1):
        super().__init__()

        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # Encoder: q_φ(z|V_p)
        self.encoder = Encoder(
            input_dim=num_classes,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )

        # Decoder: p_θ(S_p|z)
        self.decoder = Decoder(
            latent_dim=latent_dim,
            output_dim=num_classes,
            hidden_dims=hidden_dims[::-1] if hidden_dims else None,  # Reverse for decoder
            dropout=dropout
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for sampling from latent distribution.

        Implements Equation 8: z = μ + σ ⊙ ε, where ε ~ N(0, I)

        Args:
            mu (torch.Tensor): Mean [batch_size, latent_dim]
            logvar (torch.Tensor): Log variance [batch_size, latent_dim]

        Returns:
            torch.Tensor: Sampled latent variable [batch_size, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def encode(self, video_prob_dist):
        """
        Encode video-anchor probability distribution.

        Args:
            video_prob_dist (torch.Tensor): Video-anchor probability [batch_size, num_classes]

        Returns:
            tuple: (z, mu, logvar)
        """
        mu, logvar = self.encoder(video_prob_dist)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        """
        Decode latent variable to text-anchor probability distribution.

        Args:
            z (torch.Tensor): Latent variable [batch_size, latent_dim]

        Returns:
            torch.Tensor: Reconstructed probability distribution [batch_size, num_classes]
        """
        return self.decoder(z)

    def forward(self, video_prob_dist):
        """
        Forward pass through VAE.

        Args:
            video_prob_dist (torch.Tensor): Video-anchor probability [batch_size, num_classes]

        Returns:
            dict: Dictionary containing:
                - recon_text_prob: Reconstructed text-anchor probability [batch_size, num_classes]
                - mu: Latent mean [batch_size, latent_dim]
                - logvar: Latent log variance [batch_size, latent_dim]
                - z: Latent variable [batch_size, latent_dim]
        """
        # Ensure input matches module dtype/device (handles mixed precision CLIP outputs)
        module_params = next(self.parameters())
        video_prob_dist = video_prob_dist.to(
            device=module_params.device,
            dtype=module_params.dtype
        )

        # Encode
        z, mu, logvar = self.encode(video_prob_dist)

        # Decode
        recon_text_prob = self.decode(z)

        return {
            'recon_text_prob': recon_text_prob,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }

    def sample(self, num_samples, device='cuda'):
        """
        Sample from the prior distribution p(z) and generate text-anchor distributions.

        Args:
            num_samples (int): Number of samples to generate
            device (str): Device to generate samples on

        Returns:
            torch.Tensor: Generated probability distributions [num_samples, num_classes]
        """
        # Sample from standard normal prior
        z = torch.randn(num_samples, self.latent_dim, device=device)

        # Decode to get probability distributions
        with torch.no_grad():
            prob_dists = self.decode(z)

        return prob_dists

    def reconstruct(self, video_prob_dist):
        """
        Reconstruct text-anchor probability from video-anchor probability.

        Args:
            video_prob_dist (torch.Tensor): Video-anchor probability

        Returns:
            torch.Tensor: Reconstructed text-anchor probability
        """
        with torch.no_grad():
            outputs = self.forward(video_prob_dist)
            return outputs['recon_text_prob']


def compute_kl_divergence(mu, logvar):
    """
    Compute KL divergence between approximate posterior and prior.

    Implements Equation 15: KL(q_φ || p(z))

    Args:
        mu (torch.Tensor): Mean of approximate posterior [batch_size, latent_dim]
        logvar (torch.Tensor): Log variance of approximate posterior [batch_size, latent_dim]

    Returns:
        torch.Tensor: KL divergence [batch_size]
    """
    # KL divergence for diagonal Gaussian
    # KL(N(μ, σ²) || N(0, I)) = 0.5 * Σ(μ² + σ² - log(σ²) - 1)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    return kl_div


def compute_reconstruction_loss(recon_prob, target_prob):
    """
    Compute reconstruction loss (cross-entropy).

    Implements Equation 12: L_rec = -Σ S_p^(k) log(Ŝ_p^(k))

    Args:
        recon_prob (torch.Tensor): Reconstructed probability [batch_size, num_classes]
        target_prob (torch.Tensor): Target probability [batch_size, num_classes]

    Returns:
        torch.Tensor: Reconstruction loss [batch_size]
    """
    # Cross-entropy loss between probability distributions
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    recon_loss = -torch.sum(target_prob * torch.log(recon_prob + eps), dim=1)

    return recon_loss
