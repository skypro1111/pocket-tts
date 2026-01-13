"""Noise scheduling and augmentation utilities for CALM training.

This module provides utilities for:
- TrigFlow noise scheduling (αt = cos(t·π/2), σt = sin(t·π/2))
- Noise augmentation for backbone transformer input
- Gaussian temperature sampling for inference
"""

import math

import torch


def trigflow_schedule(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute TrigFlow noise schedule coefficients.

    Following the TrigFlow formulation with T = π/2:
        αt = cos(t * π/2)
        σt = sin(t * π/2)

    At t=0: α=1, σ=0 (clean signal)
    At t=1: α=0, σ=1 (pure noise)

    Args:
        t: Timestep tensor with values in [0, 1].

    Returns:
        Tuple of (alpha_t, sigma_t) tensors with same shape as t.
    """
    t_scaled = t * (math.pi / 2)
    alpha_t = torch.cos(t_scaled)
    sigma_t = torch.sin(t_scaled)
    return alpha_t, sigma_t


def add_noise_to_latents(
    x: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add noise to latents according to TrigFlow schedule.

    Computes x_t = αt * x + σt * ε where αt = cos(t*π/2), σt = sin(t*π/2).

    Args:
        x: Clean latent tensor of shape [B, S, D] or [B, D].
        t: Timestep tensor of shape [B] with values in [0, 1].
        noise: Optional noise tensor of same shape as x. If None, sampled from N(0,I).

    Returns:
        Noisy latent tensor of same shape as x.
    """
    if noise is None:
        noise = torch.randn_like(x)

    alpha_t, sigma_t = trigflow_schedule(t)

    # Broadcast timestep to match latent dimensions
    while alpha_t.dim() < x.dim():
        alpha_t = alpha_t.unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1)

    return alpha_t * x + sigma_t * noise


class BackboneNoiseAugmentation:
    """Noise augmentation for backbone transformer input.

    From the paper (Section 4.2): "We add noise to the input latent sequence
    before it enters the backbone Transformer. This noise injection encourages
    the backbone to focus on coarse structure rather than fine details."

    The noise level is sampled per-sequence and applied to all latent frames
    in that sequence.

    Args:
        min_noise_level: Minimum noise level (t value). Default 0.0.
        max_noise_level: Maximum noise level (t value). Default 0.5.
        prob: Probability of applying noise augmentation. Default 1.0.
    """

    def __init__(
        self,
        min_noise_level: float = 0.0,
        max_noise_level: float = 0.5,
        prob: float = 1.0,
    ):
        self.min_noise_level = min_noise_level
        self.max_noise_level = max_noise_level
        self.prob = prob

    def __call__(
        self,
        x: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply noise augmentation to latent sequence.

        Args:
            x: Clean latent tensor of shape [B, S, D].
            noise: Optional pre-sampled noise tensor.

        Returns:
            Tuple of (noisy_latents, noise_levels) where:
            - noisy_latents has shape [B, S, D]
            - noise_levels has shape [B] with the applied t values
        """
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        # Sample noise levels uniformly in [min, max]
        t = torch.rand(batch_size, device=device, dtype=dtype)
        t = self.min_noise_level + t * (self.max_noise_level - self.min_noise_level)

        # Optionally skip noise augmentation with probability (1 - prob)
        if self.prob < 1.0:
            mask = torch.rand(batch_size, device=device) < self.prob
            t = t * mask.float()

        # Apply noise
        noisy_x = add_noise_to_latents(x, t, noise)

        return noisy_x, t


class GaussianTemperatureSampler:
    """Gaussian temperature sampling for inference.

    From Section 4.5 of the paper:
    "To replicate temperature sampling in the continuous domain, we introduce
    a sampling heuristic... we apply temperature only on the Gaussian noise
    sampling (which is equivalent to sampling from a Gaussian with variance
    1/√τ) and get significant improvements in metrics."

    Args:
        temperature: Sampling temperature τ. Default 1.0 (standard sampling).
            - τ < 1.0: More deterministic, lower diversity
            - τ > 1.0: More random, higher diversity
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    @property
    def std(self) -> float:
        """Standard deviation of the temperature-scaled Gaussian."""
        return 1.0 / (self.temperature ** 0.5) if self.temperature > 0 else 0.0

    def sample(self, shape: tuple, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Sample from temperature-scaled Gaussian.

        Args:
            shape: Shape of tensor to sample.
            device: Device to create tensor on.
            dtype: Data type of tensor.

        Returns:
            Sampled tensor from N(0, 1/√τ).
        """
        noise = torch.randn(shape, device=device, dtype=dtype)
        return noise * self.std


def normalize_latents(
    x: torch.Tensor,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize latent vectors (center and scale).

    From the paper: "Center and normalize" is applied to audio embeddings
    before entering the backbone.

    Args:
        x: Latent tensor of shape [B, S, D] or [B, D].
        mean: Optional pre-computed mean of shape [D].
        std: Optional pre-computed std of shape [D].
        eps: Small value for numerical stability.

    Returns:
        Tuple of (normalized_x, mean, std).
    """
    if mean is None:
        # Compute mean over batch and sequence dimensions
        if x.dim() == 3:
            mean = x.mean(dim=(0, 1))
        else:
            mean = x.mean(dim=0)

    if std is None:
        # Compute std over batch and sequence dimensions
        if x.dim() == 3:
            std = x.std(dim=(0, 1))
        else:
            std = x.std(dim=0)

    std = std.clamp(min=eps)
    normalized = (x - mean) / std

    return normalized, mean, std


def denormalize_latents(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Denormalize latent vectors.

    Inverse of normalize_latents.

    Args:
        x: Normalized latent tensor.
        mean: Mean used for normalization.
        std: Std used for normalization.

    Returns:
        Denormalized tensor.
    """
    return x * std + mean


class EMAStatistics:
    """Exponential moving average for tracking latent statistics.

    Used to maintain running estimates of mean and std for normalization
    during training.

    Args:
        dim: Dimension of latent vectors.
        decay: EMA decay factor (higher = slower update). Default 0.999.
        device: Device for tensors.
        dtype: Data type for tensors.
    """

    def __init__(
        self,
        dim: int,
        decay: float = 0.999,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.decay = decay
        self.mean = torch.zeros(dim, device=device, dtype=dtype)
        self.std = torch.ones(dim, device=device, dtype=dtype)
        self.initialized = False

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update statistics with new batch of latents.

        Args:
            x: Latent tensor of shape [B, S, D] or [B, D].
        """
        # Compute batch statistics
        if x.dim() == 3:
            batch_mean = x.mean(dim=(0, 1))
            batch_std = x.std(dim=(0, 1))
        else:
            batch_mean = x.mean(dim=0)
            batch_std = x.std(dim=0)

        if not self.initialized:
            self.mean = batch_mean
            self.std = batch_std
            self.initialized = True
        else:
            self.mean = self.decay * self.mean + (1 - self.decay) * batch_mean
            self.std = self.decay * self.std + (1 - self.decay) * batch_std

    def normalize(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Normalize using current EMA statistics.

        Args:
            x: Latent tensor to normalize.
            eps: Small value for numerical stability.

        Returns:
            Normalized tensor.
        """
        std = self.std.clamp(min=eps)
        return (x - self.mean) / std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize using current EMA statistics.

        Args:
            x: Normalized latent tensor.

        Returns:
            Denormalized tensor.
        """
        return x * self.std + self.mean

    def to(self, device: torch.device) -> "EMAStatistics":
        """Move statistics to device."""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self
