"""
rectified_flow.py — Rectified Flow training objective and ODE sampling.

Core identity: x_t = (1 - t) * x_0 + t * x_1
              v_target = x_1 - x_0      (constant velocity along the straight line)

The model learns v_θ(x_t, t) ≈ v_target = x_1 - x_0
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class RectifiedFlowTrainer:
    """
    Training helper for Rectified Flow.

    The loss is simply:
        L = E_{t, x_0, x_1} || v_θ(x_t, t) - (x_1 - x_0) ||^2

    where x_t = (1 - t) * x_0 + t * x_1, with x_0 ~ N(0, I), x_1 ~ p_data.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def compute_loss(self, x_1: torch.Tensor) -> torch.Tensor:
        """
        Compute the RF training loss for a batch of data x_1.

        Args:
            x_1: clean data batch [B, C, H, W]
        Returns:
            loss: scalar MSE loss
        """
        B = x_1.shape[0]
        device = x_1.device

        # Sample noise (source distribution)
        x_0 = torch.randn_like(x_1)

        # Sample time uniformly
        t = torch.rand(B, device=device)

        # Interpolate: x_t = (1 - t) * x_0 + t * x_1
        t_expand = t[:, None, None, None]
        x_t = (1 - t_expand) * x_0 + t_expand * x_1

        # Target velocity: v = x_1 - x_0 (the straight-line direction)
        v_target = x_1 - x_0

        # Predicted velocity
        v_pred = self.model(x_t, t)

        # MSE loss
        loss = ((v_pred - v_target) ** 2).mean()
        return loss


class RectifiedFlowSampler:
    """
    ODE sampler for Rectified Flow using Euler method.

    dx_t = v_θ(x_t, t) dt

    Supports configurable step sizes to simulate edge (large Δt) vs cloud (small Δt).
    """

    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad()
    def sample(
        self,
        x_0: torch.Tensor,
        num_steps: int = 50,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate samples by integrating from t_start to t_end.

        Args:
            x_0:       initial noise [B, C, H, W] (or intermediate state if t_start > 0)
            num_steps: number of Euler steps
            t_start:   starting time (0 = pure noise)
            t_end:     ending time (1 = clean image)
        Returns:
            x:         final state at t_end
        """
        dt = (t_end - t_start) / num_steps
        x = x_0.clone()
        t = t_start

        for step in range(num_steps):
            t_batch = torch.full((x.shape[0],), t, device=x.device)
            v = self.model(x, t_batch)
            x = x + v * dt
            t += dt

        return x

    @torch.no_grad()
    def sample_with_trajectory(
        self,
        x_0: torch.Tensor,
        num_steps: int = 50,
        t_start: float = 0.0,
        t_end: float = 1.0,
        record_every: int = 1,
    ) -> tuple:
        """
        Sample while recording the full trajectory for visualization.

        Returns:
            x_final:    final state
            trajectory: list of (t, x_t) tuples
        """
        dt = (t_end - t_start) / num_steps
        x = x_0.clone()
        t = t_start
        trajectory = [(t, x.clone())]

        for step in range(num_steps):
            t_batch = torch.full((x.shape[0],), t, device=x.device)
            v = self.model(x, t_batch)
            x = x + v * dt
            t += dt
            if (step + 1) % record_every == 0:
                trajectory.append((t, x.clone()))

        return x, trajectory

    @torch.no_grad()
    def get_velocity(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Get the velocity field at a specific (x, t) point."""
        t_batch = torch.full((x.shape[0],), t, device=x.device)
        return self.model(x, t_batch)


# ============================================================
# Direction Correction Collaborative Sampler
# ============================================================

class DirectionCorrectionSampler:
    """
    The core of our method: edge-cloud collaborative inference via direction correction.

    Protocol:
    1. Edge runs from t=0 to t=t*, producing x_{t*}^edge
    2. Cloud computes v_cloud at (x_{t*}^edge, t*)
    3. Edge computes v_edge at (x_{t*}^edge, t*)
    4. δv = v_cloud - v_edge is transmitted back (highly compressible)
    5. Edge continues from t* to 1, injecting α(t) · δv into its velocity
    """

    def __init__(self, edge_model: nn.Module, cloud_model: nn.Module):
        self.edge_sampler = RectifiedFlowSampler(edge_model)
        self.cloud_sampler = RectifiedFlowSampler(cloud_model)
        self.edge_model = edge_model
        self.cloud_model = cloud_model

    @torch.no_grad()
    def sample_direction_correction(
        self,
        x_0: torch.Tensor,
        t_star: float = 0.5,
        edge_steps_before: int = 10,
        edge_steps_after: int = 10,
        compress_fn=None,
    ) -> dict:
        """
        Full direction correction protocol.

        Args:
            x_0:               initial noise [B, C, H, W]
            t_star:            split point in [0, 1]
            edge_steps_before: Euler steps for edge from 0 to t*
            edge_steps_after:  Euler steps for edge from t* to 1
            compress_fn:       optional compression function for δv

        Returns:
            dict with keys:
                'x_final':       final generated image
                'delta_v':       raw direction correction vector
                'delta_v_compressed': compressed δv (if compress_fn given)
                'alpha_values':  α(t) gate values over time
                'x_t_star':      intermediate state at t*
        """
        device = x_0.device

        # ---- Phase 1: Edge runs from 0 to t* ----
        x_t_star = self.edge_sampler.sample(
            x_0, num_steps=edge_steps_before, t_start=0.0, t_end=t_star
        )

        # ---- Phase 2: Compute direction correction δv ----
        t_batch = torch.full((x_0.shape[0],), t_star, device=device)
        v_edge = self.edge_model(x_t_star, t_batch)
        v_cloud = self.cloud_model(x_t_star, t_batch)
        delta_v = v_cloud - v_edge

        # Optional compression
        delta_v_used = delta_v
        delta_v_compressed = None
        if compress_fn is not None:
            delta_v_compressed = compress_fn(delta_v)
            delta_v_used = delta_v_compressed

        # ---- Phase 3: Edge continues from t* to 1 with correction ----
        dt = (1.0 - t_star) / edge_steps_after
        x = x_t_star.clone()
        t = t_star

        alpha_values = []

        for step in range(edge_steps_after):
            t_batch = torch.full((x.shape[0],), t, device=device)
            v_current = self.edge_model(x, t_batch)

            # Full correction: RF velocity is approximately constant along each
            # trajectory, so δv(t*) is a valid correction for all t > t*
            v_corrected = v_current + delta_v_used
            alpha_values.append(1.0)

            x = x + v_corrected * dt
            t += dt

        return {
            'x_final': x,
            'delta_v': delta_v,
            'delta_v_compressed': delta_v_compressed,
            'alpha_values': alpha_values,
            'x_t_star': x_t_star,
        }

    @torch.no_grad()
    def sample_state_relay(
        self,
        x_0: torch.Tensor,
        t_star: float = 0.5,
        edge_steps: int = 10,
        cloud_steps: int = 25,
    ) -> torch.Tensor:
        """
        Baseline: state relay. Edge runs to t*, transmits x_{t*}, cloud finishes.
        This is what Hybrid SD and similar methods do.
        """
        # Edge runs from 0 to t*
        x_t_star = self.edge_sampler.sample(
            x_0, num_steps=edge_steps, t_start=0.0, t_end=t_star
        )
        # Cloud continues from t* to 1
        x_final = self.cloud_sampler.sample(
            x_t_star, num_steps=cloud_steps, t_start=t_star, t_end=1.0
        )
        return x_final

    @torch.no_grad()
    def sample_edge_only(
        self,
        x_0: torch.Tensor,
        num_steps: int = 20,
    ) -> torch.Tensor:
        """Baseline: edge model runs the entire path alone."""
        return self.edge_sampler.sample(x_0, num_steps=num_steps, t_start=0.0, t_end=1.0)

    @torch.no_grad()
    def sample_cloud_only(
        self,
        x_0: torch.Tensor,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """Reference: cloud model runs the entire path (upper bound on quality)."""
        return self.cloud_sampler.sample(x_0, num_steps=num_steps, t_start=0.0, t_end=1.0)

    @staticmethod
    def _compute_alpha(v_current: torch.Tensor, v_ref: torch.Tensor) -> torch.Tensor:
        """
        Adaptive gate: cosine similarity between current and reference velocity.
        Returns per-sample α ∈ [-1, 1], clamped to [0, 1].

        α(t) = cos_sim(v_edge(x_t, t), v_edge(x_{t*}, t*))
        """
        B = v_current.shape[0]
        v1 = v_current.reshape(B, -1)
        v2 = v_ref.reshape(B, -1)
        cos_sim = F.cosine_similarity(v1, v2, dim=-1)
        return cos_sim.clamp(min=0.0)  # only positive corrections


# Need F for cosine_similarity
import torch.nn.functional as F
