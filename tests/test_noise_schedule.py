"""
Tests for graph_diffusion.building_blocks.noise_schedule
=========================================================

Unit tests for NoiseSchedule with linear and cosine schedules.
"""

import torch
from context import graph_diffusion  # noqa: F401

from graph_diffusion.building_blocks.noise_schedule import NoiseSchedule


class TestNoiseScheduleLinear:
    def test_buffer_shapes(self):
        """All buffers should have shape (T,) for the linear schedule."""
        torch.manual_seed(0)
        num_steps = 100
        ns = NoiseSchedule(T=num_steps, schedule_type="linear")
        for name in [
            "betas",
            "alphas",
            "alphas_cumprod",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
        ]:
            buf = getattr(ns, name)
            assert buf.shape == (num_steps,), f"{name} has wrong shape: {buf.shape}"

    def test_betas_range(self):
        """Betas should be in (0, 1) for the linear schedule."""
        torch.manual_seed(0)
        ns = NoiseSchedule(T=1000, schedule_type="linear")
        assert (ns.betas > 0).all()
        assert (ns.betas < 1).all()

    def test_alphas_cumprod_decreasing(self):
        """alphas_cumprod should be monotonically decreasing."""
        torch.manual_seed(0)
        ns = NoiseSchedule(T=1000, schedule_type="linear")
        diffs = ns.alphas_cumprod[1:] - ns.alphas_cumprod[:-1]
        assert (diffs < 0).all()

    def test_alphas_plus_betas(self):
        """alphas + betas should equal 1."""
        torch.manual_seed(0)
        ns = NoiseSchedule(T=100, schedule_type="linear")
        assert torch.allclose(ns.alphas + ns.betas, torch.ones(100))

    def test_sqrt_consistency(self):
        """sqrt buffers should be consistent with alphas_cumprod."""
        torch.manual_seed(0)
        ns = NoiseSchedule(T=100, schedule_type="linear")
        assert torch.allclose(ns.sqrt_alphas_cumprod, torch.sqrt(ns.alphas_cumprod))
        assert torch.allclose(
            ns.sqrt_one_minus_alphas_cumprod,
            torch.sqrt(1.0 - ns.alphas_cumprod),
        )


class TestNoiseScheduleCosine:
    def test_buffer_shapes(self):
        """All buffers should have shape (T,) for the cosine schedule."""
        torch.manual_seed(0)
        num_steps = 200
        ns = NoiseSchedule(T=num_steps, schedule_type="cosine")
        for name in [
            "betas",
            "alphas",
            "alphas_cumprod",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
        ]:
            buf = getattr(ns, name)
            assert buf.shape == (num_steps,), f"{name} has wrong shape: {buf.shape}"

    def test_betas_clamped(self):
        """Cosine betas should be clamped to max 0.999."""
        torch.manual_seed(0)
        ns = NoiseSchedule(T=1000, schedule_type="cosine")
        assert (ns.betas <= 0.999).all()
        assert (ns.betas > 0).all()

    def test_alphas_cumprod_decreasing(self):
        """alphas_cumprod should be monotonically decreasing."""
        torch.manual_seed(0)
        ns = NoiseSchedule(T=1000, schedule_type="cosine")
        diffs = ns.alphas_cumprod[1:] - ns.alphas_cumprod[:-1]
        assert (diffs < 0).all()


class TestNoiseScheduleGetT:
    def test_output_shape(self):
        """get_t should return shape (B, 1)."""
        torch.manual_seed(0)
        ns = NoiseSchedule(T=100, schedule_type="linear")
        t = torch.tensor([0, 10, 50, 99])
        out = ns.get_t(t, "sqrt_alphas_cumprod")
        assert out.shape == (4, 1)

    def test_single_timestep(self):
        """get_t with batch size 1 should return (1, 1)."""
        torch.manual_seed(0)
        ns = NoiseSchedule(T=100, schedule_type="linear")
        t = torch.tensor([5])
        out = ns.get_t(t, "betas")
        assert out.shape == (1, 1)

    def test_correct_values(self):
        """get_t should return the correct indexed values."""
        torch.manual_seed(0)
        ns = NoiseSchedule(T=100, schedule_type="linear")
        t = torch.tensor([0, 99])
        out = ns.get_t(t, "betas")
        assert torch.allclose(out[0, 0], ns.betas[0])
        assert torch.allclose(out[1, 0], ns.betas[99])

    def test_invalid_buffer_name(self):
        """get_t should raise ValueError for unknown buffer names."""
        torch.manual_seed(0)
        ns = NoiseSchedule(T=100, schedule_type="linear")
        t = torch.tensor([0])
        try:
            ns.get_t(t, "nonexistent_buffer")
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass


class TestNoiseScheduleValidation:
    def test_invalid_schedule_type(self):
        """Unknown schedule_type should raise ValueError."""
        try:
            NoiseSchedule(schedule_type="exponential")
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_invalid_num_steps(self):  # noqa: N802
        """T < 1 should raise ValueError."""
        try:
            NoiseSchedule(T=0)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_buffers_are_float32(self):
        """All buffers should be float32."""
        torch.manual_seed(0)
        ns = NoiseSchedule(T=50, schedule_type="cosine")
        for name in ["betas", "alphas", "alphas_cumprod"]:
            assert getattr(ns, name).dtype == torch.float32
