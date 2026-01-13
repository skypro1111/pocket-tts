"""Tests for CALM loss functions and noise schedule utilities."""

import math

import pytest
import torch

from training.utils.losses import (
    AdaptiveWeightFunction,
    CALMLoss,
    ConsistencyModelLoss,
    EOSLoss,
    FlowMatchingLoss,
    interpolate_latents,
)
from training.utils.noise_schedule import (
    BackboneNoiseAugmentation,
    EMAStatistics,
    GaussianTemperatureSampler,
    add_noise_to_latents,
    trigflow_schedule,
)


class TestTrigFlowSchedule:
    """Tests for TrigFlow noise schedule."""

    def test_schedule_at_t0(self):
        """At t=0, alpha=1 and sigma=0 (clean signal)."""
        t = torch.tensor([0.0])
        alpha, sigma = trigflow_schedule(t)
        assert torch.allclose(alpha, torch.tensor([1.0]))
        assert torch.allclose(sigma, torch.tensor([0.0]))

    def test_schedule_at_t1(self):
        """At t=1, alpha=0 and sigma=1 (pure noise)."""
        t = torch.tensor([1.0])
        alpha, sigma = trigflow_schedule(t)
        assert torch.allclose(alpha, torch.tensor([0.0]), atol=1e-6)
        assert torch.allclose(sigma, torch.tensor([1.0]))

    def test_schedule_at_t05(self):
        """At t=0.5, alpha=sigma=sqrt(2)/2."""
        t = torch.tensor([0.5])
        alpha, sigma = trigflow_schedule(t)
        expected = math.cos(0.5 * math.pi / 2)
        assert torch.allclose(alpha, torch.tensor([expected]))
        assert torch.allclose(sigma, torch.tensor([math.sin(0.5 * math.pi / 2)]))

    def test_schedule_batch(self):
        """Test with batch of timesteps."""
        t = torch.tensor([0.0, 0.5, 1.0])
        alpha, sigma = trigflow_schedule(t)
        assert alpha.shape == (3,)
        assert sigma.shape == (3,)


class TestInterpolateLatents:
    """Tests for latent interpolation."""

    def test_interpolate_at_t0(self):
        """At t=0, should return clean latents."""
        x_clean = torch.randn(2, 10, 32)
        noise = torch.randn(2, 10, 32)
        t = torch.tensor([0.0, 0.0])

        result = interpolate_latents(x_clean, noise, t)
        assert torch.allclose(result, x_clean, atol=1e-5)

    def test_interpolate_at_t1(self):
        """At t=1, should return noise."""
        x_clean = torch.randn(2, 10, 32)
        noise = torch.randn(2, 10, 32)
        t = torch.tensor([1.0, 1.0])

        result = interpolate_latents(x_clean, noise, t)
        assert torch.allclose(result, noise, atol=1e-5)

    def test_interpolate_output_shape(self):
        """Output shape should match input shape."""
        x_clean = torch.randn(4, 20, 64)
        noise = torch.randn(4, 20, 64)
        t = torch.rand(4)

        result = interpolate_latents(x_clean, noise, t.view(4, 1, 1))
        assert result.shape == x_clean.shape


class TestAdaptiveWeightFunction:
    """Tests for adaptive weight function."""

    def test_output_shape(self):
        """Output should have same shape as input timesteps."""
        weight_fn = AdaptiveWeightFunction(hidden_dim=32, num_layers=2)
        t = torch.rand(10)
        w = weight_fn(t)
        assert w.shape == (10,)

    def test_scalar_input(self):
        """Should handle scalar input."""
        weight_fn = AdaptiveWeightFunction(hidden_dim=32, num_layers=2)
        t = torch.tensor(0.5)
        w = weight_fn(t)
        assert w.dim() == 0 or w.shape == ()

    def test_learnable_parameters(self):
        """Weight function should have learnable parameters."""
        weight_fn = AdaptiveWeightFunction(hidden_dim=32, num_layers=2)
        num_params = sum(p.numel() for p in weight_fn.parameters())
        assert num_params > 0


class TestConsistencyModelLoss:
    """Tests for consistency model loss."""

    def test_loss_computation(self):
        """Test that loss can be computed without errors."""
        # Create a simple mock flow network
        class MockFlowNet(torch.nn.Module):
            def __init__(self, latent_dim, cond_dim):
                super().__init__()
                self.linear = torch.nn.Linear(latent_dim + cond_dim + 2, latent_dim)

            def forward(self, c, s, t, x):
                combined = torch.cat([c, s, t, x], dim=-1)
                return self.linear(combined)

        batch_size, seq_len, latent_dim, cond_dim = 2, 10, 32, 64
        flow_net = MockFlowNet(latent_dim, cond_dim)
        loss_fn = ConsistencyModelLoss(adaptive_weight=False)

        conditioning = torch.randn(batch_size, seq_len, cond_dim)
        x_clean = torch.randn(batch_size, seq_len, latent_dim)

        loss, metrics = loss_fn(flow_net, conditioning, x_clean)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert "consistency_loss" in metrics
        assert "sq_diff_mean" in metrics

    def test_loss_with_adaptive_weight(self):
        """Test loss with adaptive weighting enabled."""

        class MockFlowNet(torch.nn.Module):
            def __init__(self, latent_dim, cond_dim):
                super().__init__()
                self.linear = torch.nn.Linear(latent_dim + cond_dim + 2, latent_dim)

            def forward(self, c, s, t, x):
                combined = torch.cat([c, s, t, x], dim=-1)
                return self.linear(combined)

        batch_size, seq_len, latent_dim, cond_dim = 2, 10, 32, 64
        flow_net = MockFlowNet(latent_dim, cond_dim)
        loss_fn = ConsistencyModelLoss(adaptive_weight=True)

        conditioning = torch.randn(batch_size, seq_len, cond_dim)
        x_clean = torch.randn(batch_size, seq_len, latent_dim)

        loss, metrics = loss_fn(flow_net, conditioning, x_clean)

        assert "weight_mean" in metrics


class TestFlowMatchingLoss:
    """Tests for flow matching loss."""

    def test_loss_computation(self):
        """Test that loss can be computed without errors."""

        class MockFlowNet(torch.nn.Module):
            def __init__(self, latent_dim, cond_dim):
                super().__init__()
                self.linear = torch.nn.Linear(latent_dim + cond_dim + 2, latent_dim)

            def forward(self, c, s, t, x):
                combined = torch.cat([c, s, t, x], dim=-1)
                return self.linear(combined)

        batch_size, seq_len, latent_dim, cond_dim = 2, 10, 32, 64
        flow_net = MockFlowNet(latent_dim, cond_dim)
        loss_fn = FlowMatchingLoss()

        conditioning = torch.randn(batch_size, seq_len, cond_dim)
        x_clean = torch.randn(batch_size, seq_len, latent_dim)

        loss, metrics = loss_fn(flow_net, conditioning, x_clean)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert "flow_matching_loss" in metrics


class TestEOSLoss:
    """Tests for EOS prediction loss."""

    def test_loss_computation(self):
        """Test EOS loss computation."""
        loss_fn = EOSLoss()

        batch_size, seq_len = 4, 20
        logits = torch.randn(batch_size, seq_len)
        targets = torch.zeros(batch_size, seq_len)
        # Set EOS at different positions
        targets[0, 10] = 1.0
        targets[1, 15] = 1.0
        targets[2, 5] = 1.0
        targets[3, 18] = 1.0

        loss, metrics = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert "eos_loss" in metrics
        assert "eos_accuracy" in metrics
        assert 0.0 <= metrics["eos_accuracy"] <= 1.0

    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        loss_fn = EOSLoss()

        # Create perfect predictions
        targets = torch.zeros(2, 10)
        targets[0, 5] = 1.0
        targets[1, 8] = 1.0

        # Logits that would produce correct predictions
        logits = torch.full((2, 10), -10.0)
        logits[0, 5] = 10.0
        logits[1, 8] = 10.0

        loss, metrics = loss_fn(logits, targets)
        # Accuracy should be high (all correct)
        assert metrics["eos_accuracy"] == 1.0


class TestCALMLoss:
    """Tests for combined CALM loss."""

    def test_combined_loss(self):
        """Test combined loss computation."""

        class MockFlowNet(torch.nn.Module):
            def __init__(self, latent_dim, cond_dim):
                super().__init__()
                self.linear = torch.nn.Linear(latent_dim + cond_dim + 2, latent_dim)

            def forward(self, c, s, t, x):
                combined = torch.cat([c, s, t, x], dim=-1)
                return self.linear(combined)

        batch_size, seq_len, latent_dim, cond_dim = 2, 10, 32, 64
        flow_net = MockFlowNet(latent_dim, cond_dim)

        loss_fn = CALMLoss(use_consistency=True, eos_weight=0.1)

        conditioning = torch.randn(batch_size, seq_len, cond_dim)
        x_clean = torch.randn(batch_size, seq_len, latent_dim)
        eos_logits = torch.randn(batch_size, seq_len)
        eos_targets = torch.zeros(batch_size, seq_len)
        eos_targets[:, -1] = 1.0

        loss, metrics = loss_fn(
            flow_net, conditioning, x_clean, eos_logits, eos_targets
        )

        assert isinstance(loss, torch.Tensor)
        assert "total_loss" in metrics
        assert "eos_loss" in metrics


class TestBackboneNoiseAugmentation:
    """Tests for backbone noise augmentation."""

    def test_augmentation_output_shape(self):
        """Output should have same shape as input."""
        aug = BackboneNoiseAugmentation(min_noise_level=0.0, max_noise_level=0.5)
        x = torch.randn(4, 20, 32)
        noisy_x, t = aug(x)
        assert noisy_x.shape == x.shape
        assert t.shape == (4,)

    def test_noise_levels_in_range(self):
        """Noise levels should be in specified range."""
        aug = BackboneNoiseAugmentation(min_noise_level=0.1, max_noise_level=0.3)
        x = torch.randn(100, 10, 16)
        _, t = aug(x)
        assert (t >= 0.1).all()
        assert (t <= 0.3).all()

    def test_prob_zero(self):
        """With prob=0, no noise should be applied."""
        aug = BackboneNoiseAugmentation(
            min_noise_level=0.0, max_noise_level=0.5, prob=0.0
        )
        x = torch.randn(4, 10, 16)
        noisy_x, t = aug(x)
        # All noise levels should be 0
        assert (t == 0).all()


class TestEMAStatistics:
    """Tests for EMA statistics tracking."""

    def test_initialization(self):
        """Test initial values."""
        ema = EMAStatistics(dim=32)
        assert ema.mean.shape == (32,)
        assert ema.std.shape == (32,)
        assert (ema.mean == 0).all()
        assert (ema.std == 1).all()

    def test_update(self):
        """Test that update changes statistics."""
        ema = EMAStatistics(dim=16, decay=0.9)
        x = torch.randn(10, 5, 16) * 2 + 3  # Non-zero mean and std

        ema.update(x)

        assert ema.initialized
        # Mean should have moved from 0
        assert not torch.allclose(ema.mean, torch.zeros(16))

    def test_normalize_denormalize(self):
        """Normalize then denormalize should recover original."""
        ema = EMAStatistics(dim=16, decay=0.9)
        x = torch.randn(4, 8, 16) * 2 + 3

        # Update with some data
        ema.update(x)

        # Create new data
        y = torch.randn(2, 4, 16)
        normalized = ema.normalize(y)
        recovered = ema.denormalize(normalized)

        assert torch.allclose(recovered, y, atol=1e-5)


class TestGaussianTemperatureSampler:
    """Tests for Gaussian temperature sampling."""

    def test_default_temperature(self):
        """Default temperature should give standard normal."""
        sampler = GaussianTemperatureSampler(temperature=1.0)
        assert sampler.std == 1.0

    def test_low_temperature(self):
        """Low temperature should give lower std."""
        sampler = GaussianTemperatureSampler(temperature=0.5)
        assert sampler.std > 1.0  # sqrt(1/0.5) = sqrt(2)

    def test_high_temperature(self):
        """High temperature should give higher std."""
        sampler = GaussianTemperatureSampler(temperature=2.0)
        assert sampler.std < 1.0  # sqrt(1/2)

    def test_sample_shape(self):
        """Sample should have correct shape."""
        sampler = GaussianTemperatureSampler(temperature=1.0)
        sample = sampler.sample((4, 10, 32), device=torch.device("cpu"), dtype=torch.float32)
        assert sample.shape == (4, 10, 32)

    def test_sample_distribution(self):
        """Sample std should match expected."""
        sampler = GaussianTemperatureSampler(temperature=0.5)
        samples = sampler.sample((10000,), device=torch.device("cpu"), dtype=torch.float32)
        actual_std = samples.std().item()
        expected_std = sampler.std
        # Allow 10% tolerance
        assert abs(actual_std - expected_std) / expected_std < 0.1
