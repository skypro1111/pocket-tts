"""Loss functions for training Pocket TTS based on CALM paper (arxiv 2509.06926).

This module implements the loss functions described in the CALM (Continuous Audio
Language Models) paper, including:
- Consistency Model Loss (LCALM) - Equation 3
- Adaptive Weighting Function
- TrigFlow/Flow Matching Loss
- EOS (End of Sequence) Prediction Loss
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveWeightFunction(nn.Module):
    """Learnable adaptive weighting function wψ(t) for consistency loss.

    This implements the adaptive weighting scheme from the continuous-time
    consistency loss (Equation 1 in the paper). The weighting function helps
    balance the loss contribution at different timesteps.

    Args:
        hidden_dim: Hidden dimension for the weighting network.
        num_layers: Number of layers in the weighting MLP.
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()

        layers = []
        input_dim = 1
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers.append(nn.Linear(input_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute adaptive weight for timestep t.

        Args:
            t: Timestep tensor of shape [...] with values in [0, 1].

        Returns:
            Weight tensor of the same shape as t.
        """
        t_input = t.unsqueeze(-1) if t.dim() == 0 or t.shape[-1] != 1 else t
        return self.mlp(t_input).squeeze(-1)


def trigflow_schedule(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute TrigFlow noise schedule coefficients.

    Following the TrigFlow formulation with T = π/2:
        αt = cos(t * π/2)
        σt = sin(t * π/2)

    Args:
        t: Timestep tensor with values in [0, 1].

    Returns:
        Tuple of (alpha_t, sigma_t) tensors.
    """
    t_scaled = t * (math.pi / 2)
    alpha_t = torch.cos(t_scaled)
    sigma_t = torch.sin(t_scaled)
    return alpha_t, sigma_t


def interpolate_latents(
    x_clean: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Interpolate between clean latents and noise at timestep t.

    Computes x_t = αt * x_clean + σt * noise using TrigFlow schedule.

    Args:
        x_clean: Clean latent tensor of shape [B, S, D].
        noise: Noise tensor of the same shape as x_clean.
        t: Timestep tensor of shape [B, 1, 1] or broadcastable.

    Returns:
        Interpolated tensor x_t of shape [B, S, D].
    """
    alpha_t, sigma_t = trigflow_schedule(t)
    # Broadcast timestep to match latent dimensions
    while alpha_t.dim() < x_clean.dim():
        alpha_t = alpha_t.unsqueeze(-1)
        sigma_t = sigma_t.unsqueeze(-1)

    return alpha_t * x_clean + sigma_t * noise


class ConsistencyModelLoss(nn.Module):
    """Continuous-time consistency model loss for CALM.

    Implements Equation 3 from the paper:

    LCALM(θ, ϕ, ψ) = Σs Et,ε [
        exp(wψ(t)) ||Fϕ(x^s_t, t, Z^s) - F̄ϕ(x^s_t, t, Z^s) - cos(t)∂f̄ϕ/∂t||²₂
        - wψ(t)
    ]

    where:
    - Fϕ is the consistency model (predicts x_1 from x_t)
    - F̄ϕ is a stop-gradient version of Fϕ
    - wψ(t) is the adaptive weighting function
    - Z^s is the combined long and short context
    - x^s_t = cos(t)x^s + sin(t)ε is the noisy latent

    Args:
        adaptive_weight: Whether to use learnable adaptive weighting.
        weight_hidden_dim: Hidden dimension for adaptive weight network.
    """

    def __init__(
        self,
        adaptive_weight: bool = True,
        weight_hidden_dim: int = 64,
    ):
        super().__init__()
        self.adaptive_weight = adaptive_weight

        if adaptive_weight:
            self.weight_fn = AdaptiveWeightFunction(hidden_dim=weight_hidden_dim)
        else:
            self.weight_fn = None

    def forward(
        self,
        flow_net: nn.Module,
        conditioning: torch.Tensor,
        x_clean: torch.Tensor,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, dict]:
        """Compute consistency model loss.

        Args:
            flow_net: The flow network (SimpleMLPAdaLN) that predicts flow direction.
            conditioning: Conditioning tensor Z^s from backbone [B, S, D].
            x_clean: Clean target latent vectors [B, S, C].
            t: Optional timestep tensor [B] in [0, 1]. If None, sampled uniformly.
            noise: Optional noise tensor [B, S, C]. If None, sampled from N(0, I).
            eps: Small value to avoid t=0 singularity.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        batch_size, seq_len, latent_dim = x_clean.shape
        device = x_clean.device
        dtype = x_clean.dtype

        # Sample timestep uniformly in [eps, 1] if not provided
        if t is None:
            t = torch.rand(batch_size, device=device, dtype=dtype) * (1 - eps) + eps

        # Sample noise if not provided
        if noise is None:
            noise = torch.randn_like(x_clean)

        # Compute noisy latents: x_t = cos(t)x + sin(t)ε
        t_expanded = t.view(batch_size, 1, 1)
        x_t = interpolate_latents(x_clean, noise, t_expanded)

        # Get s (start time) and t (target time) for the flow network
        # For consistency models, we predict the flow from s to t where s < t
        # Here we use s = 0 (starting from pure noise conceptually)
        s_tensor = torch.zeros_like(t)

        # Expand timesteps for per-position prediction
        s_expanded = s_tensor.view(batch_size, 1, 1).expand(-1, seq_len, 1)
        t_full = t_expanded.expand(-1, seq_len, 1)

        # Flatten for MLP processing: [B*S, ...]
        x_t_flat = x_t.view(-1, latent_dim)
        cond_flat = conditioning.view(-1, conditioning.shape[-1])
        s_flat = s_expanded.reshape(-1, 1)
        t_flat = t_full.reshape(-1, 1)

        # Forward pass through flow network: predicts flow direction
        # flow_net(c, s, t, x) -> direction
        flow_pred = flow_net(cond_flat, s_flat, t_flat, x_t_flat)
        flow_pred = flow_pred.view(batch_size, seq_len, latent_dim)

        # Compute consistency target with stop-gradient
        with torch.no_grad():
            flow_pred_sg = flow_net(cond_flat, s_flat, t_flat, x_t_flat)
            flow_pred_sg = flow_pred_sg.view(batch_size, seq_len, latent_dim)

            # Compute derivative term: cos(t) * df/dt
            # Numerical approximation with small delta
            delta = 0.01
            t_plus = torch.clamp(t + delta, max=1.0)
            t_plus_expanded = t_plus.view(batch_size, 1, 1)
            t_plus_full = t_plus_expanded.expand(-1, seq_len, 1).reshape(-1, 1)

            flow_pred_plus = flow_net(cond_flat, s_flat, t_plus_full, x_t_flat)
            flow_pred_plus = flow_pred_plus.view(batch_size, seq_len, latent_dim)

            # Derivative approximation
            alpha_t, _ = trigflow_schedule(t)
            cos_t = alpha_t.view(batch_size, 1, 1)
            df_dt = (flow_pred_plus - flow_pred_sg) / delta
            derivative_term = cos_t * df_dt

            target = flow_pred_sg + derivative_term

        # Compute squared difference
        diff = flow_pred - target
        sq_diff = (diff ** 2).sum(dim=-1)  # [B, S]

        # Apply adaptive weighting
        if self.adaptive_weight and self.weight_fn is not None:
            w_t = self.weight_fn(t)  # [B]
            exp_w = torch.exp(w_t).unsqueeze(-1)  # [B, 1]
            weighted_loss = exp_w * sq_diff - w_t.unsqueeze(-1)
        else:
            weighted_loss = sq_diff

        # Average over sequence and batch
        loss = weighted_loss.mean()

        # Compute metrics
        metrics = {
            "consistency_loss": loss.item(),
            "sq_diff_mean": sq_diff.mean().item(),
            "t_mean": t.mean().item(),
        }
        if self.adaptive_weight and self.weight_fn is not None:
            metrics["weight_mean"] = w_t.mean().item()

        return loss, metrics


class FlowMatchingLoss(nn.Module):
    """Flow matching loss for training continuous latent models.

    This implements the diffusion/flow matching objective from MAR:
    L_diff(θ, ϕ) = Σs E_ε,t [||ε - εϕ(x^s_t, z^s, t)||²]

    where x^s_t = αt·x^s + σt·ε is the noisy latent at timestep t.

    This is a simpler alternative to the consistency loss that requires
    more inference steps but is easier to train.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        flow_net: nn.Module,
        conditioning: torch.Tensor,
        x_clean: torch.Tensor,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, dict]:
        """Compute flow matching loss.

        Args:
            flow_net: The flow network that predicts noise/velocity.
            conditioning: Conditioning tensor Z^s from backbone [B, S, D].
            x_clean: Clean target latent vectors [B, S, C].
            t: Optional timestep tensor [B] in [0, 1]. If None, sampled uniformly.
            noise: Optional noise tensor [B, S, C]. If None, sampled from N(0, I).
            eps: Small value to avoid t=0 singularity.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        batch_size, seq_len, latent_dim = x_clean.shape
        device = x_clean.device
        dtype = x_clean.dtype

        # Sample timestep uniformly in [eps, 1] if not provided
        if t is None:
            t = torch.rand(batch_size, device=device, dtype=dtype) * (1 - eps) + eps

        # Sample noise if not provided
        if noise is None:
            noise = torch.randn_like(x_clean)

        # Compute noisy latents: x_t = cos(t)x + sin(t)ε
        t_expanded = t.view(batch_size, 1, 1)
        x_t = interpolate_latents(x_clean, noise, t_expanded)

        # For flow matching, the target is the velocity field
        # v(x_t, t) = d/dt(αt·x + σt·ε) = -sin(t)·π/2·x + cos(t)·π/2·ε
        alpha_t, sigma_t = trigflow_schedule(t)
        alpha_t = alpha_t.view(batch_size, 1, 1)
        sigma_t = sigma_t.view(batch_size, 1, 1)

        # Velocity target (derivative of interpolation)
        velocity_target = -sigma_t * (math.pi / 2) * x_clean + alpha_t * (math.pi / 2) * noise

        # Prepare inputs for flow network
        s_tensor = torch.zeros_like(t)
        s_expanded = s_tensor.view(batch_size, 1, 1).expand(-1, seq_len, 1)
        t_full = t_expanded.expand(-1, seq_len, 1)

        # Flatten for MLP processing
        x_t_flat = x_t.view(-1, latent_dim)
        cond_flat = conditioning.view(-1, conditioning.shape[-1])
        s_flat = s_expanded.reshape(-1, 1)
        t_flat = t_full.reshape(-1, 1)

        # Predict velocity
        velocity_pred = flow_net(cond_flat, s_flat, t_flat, x_t_flat)
        velocity_pred = velocity_pred.view(batch_size, seq_len, latent_dim)

        # MSE loss between predicted and target velocity
        loss = F.mse_loss(velocity_pred, velocity_target)

        metrics = {
            "flow_matching_loss": loss.item(),
            "velocity_norm": velocity_target.norm(dim=-1).mean().item(),
            "pred_norm": velocity_pred.norm(dim=-1).mean().item(),
            "t_mean": t.mean().item(),
        }

        return loss, metrics


class EOSLoss(nn.Module):
    """Binary cross-entropy loss for end-of-sequence prediction.

    The model predicts when to stop generating audio frames.
    """

    def __init__(self, pos_weight: float = 1.0):
        """Initialize EOS loss.

        Args:
            pos_weight: Weight for positive class (EOS=1) to handle class imbalance.
        """
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        eos_logits: torch.Tensor,
        eos_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute EOS prediction loss.

        Args:
            eos_logits: Predicted EOS logits [B, S] or [B, S, 1].
            eos_targets: Target EOS labels [B, S] (1 at sequence end, 0 elsewhere).

        Returns:
            Tuple of (loss, metrics_dict).
        """
        if eos_logits.dim() == 3:
            eos_logits = eos_logits.squeeze(-1)

        # Binary cross-entropy with logits
        pos_weight = torch.tensor(self.pos_weight, device=eos_logits.device)
        loss = F.binary_cross_entropy_with_logits(
            eos_logits,
            eos_targets.float(),
            pos_weight=pos_weight,
        )

        # Compute accuracy
        with torch.no_grad():
            preds = (eos_logits > 0).float()
            accuracy = (preds == eos_targets).float().mean()

        metrics = {
            "eos_loss": loss.item(),
            "eos_accuracy": accuracy.item(),
        }

        return loss, metrics


class CALMLoss(nn.Module):
    """Combined loss for CALM training.

    Combines:
    - Consistency model loss (main generation objective)
    - EOS prediction loss

    Args:
        use_consistency: If True, use consistency loss; otherwise use flow matching.
        eos_weight: Weight for EOS loss term.
        adaptive_weight: Whether to use adaptive weighting in consistency loss.
    """

    def __init__(
        self,
        use_consistency: bool = True,
        eos_weight: float = 0.1,
        adaptive_weight: bool = True,
    ):
        super().__init__()
        self.eos_weight = eos_weight

        if use_consistency:
            self.main_loss = ConsistencyModelLoss(adaptive_weight=adaptive_weight)
        else:
            self.main_loss = FlowMatchingLoss()

        self.eos_loss = EOSLoss()

    def forward(
        self,
        flow_net: nn.Module,
        conditioning: torch.Tensor,
        x_clean: torch.Tensor,
        eos_logits: torch.Tensor | None = None,
        eos_targets: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute combined CALM loss.

        Args:
            flow_net: The flow network (SimpleMLPAdaLN).
            conditioning: Conditioning tensor from backbone [B, S, D].
            x_clean: Clean target latent vectors [B, S, C].
            eos_logits: Optional EOS prediction logits [B, S].
            eos_targets: Optional EOS target labels [B, S].
            t: Optional timestep tensor [B].
            noise: Optional noise tensor [B, S, C].

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        # Main generation loss
        main_loss, main_metrics = self.main_loss(
            flow_net=flow_net,
            conditioning=conditioning,
            x_clean=x_clean,
            t=t,
            noise=noise,
        )

        total_loss = main_loss
        metrics = main_metrics

        # EOS loss if provided
        if eos_logits is not None and eos_targets is not None:
            eos_loss, eos_metrics = self.eos_loss(eos_logits, eos_targets)
            total_loss = total_loss + self.eos_weight * eos_loss
            metrics.update(eos_metrics)

        metrics["total_loss"] = total_loss.item()

        return total_loss, metrics
