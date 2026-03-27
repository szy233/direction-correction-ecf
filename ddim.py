"""
ddim.py — DDIM (Denoising Diffusion Implicit Models) training and sampling.

Used as a controlled comparison against Rectified Flow to demonstrate
that Direction Correction depends on RF's linear trajectory property.

Noise schedule: cosine schedule (Nichol & Dhariwal 2021)
Training objective: ε-prediction  L = ||ε_θ(x_t, t) - ε||²
Sampling: DDIM (η=0, deterministic ODE)

The same UNet architecture (models.py) is reused — only the training
objective and sampling algorithm differ.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


# ============================================================
# Noise schedule
# ============================================================

def cosine_beta_schedule(num_timesteps: int, s: float = 0.008):
    """
    Cosine schedule as proposed in "Improved DDPM" (Nichol & Dhariwal 2021).
    Returns alpha_bar (cumulative product of 1-beta) for each timestep.
    """
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps) / num_timesteps
    alpha_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    betas = betas.clamp(max=0.999)
    return betas, alpha_bar[:-1]


class DiffusionSchedule:
    """Precomputed diffusion schedule quantities."""

    def __init__(self, num_timesteps: int = 1000):
        self.T = num_timesteps
        betas, alpha_bar = cosine_beta_schedule(num_timesteps)
        self.betas = betas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        return self

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """Forward diffusion: q(x_t | x_0) = N(√ᾱ_t x_0, (1-ᾱ_t)I)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return sqrt_ab * x_0 + sqrt_1mab * noise


# ============================================================
# DDIM Trainer
# ============================================================

class DDIMTrainer:
    """
    Training helper for DDIM (ε-prediction).

    Loss: L = E_{t, x_0, ε} ||ε_θ(x_t, t/T) - ε||²

    Note: the UNet takes t ∈ [0, 1] as input (same interface as RF),
    so we normalize the discrete timestep by T.
    """

    def __init__(self, model: nn.Module, num_timesteps: int = 1000):
        self.model = model
        self.schedule = DiffusionSchedule(num_timesteps)
        self.T = num_timesteps

    def compute_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        B = x_0.shape[0]
        device = x_0.device
        self.schedule.to(device)

        # Sample random timestep
        t = torch.randint(0, self.T, (B,), device=device)

        # Sample noise and create noisy input
        noise = torch.randn_like(x_0)
        x_t = self.schedule.q_sample(x_0, t, noise)

        # Model predicts noise; input t normalized to [0, 1]
        t_normalized = t.float() / self.T
        noise_pred = self.model(x_t, t_normalized)

        # MSE loss
        loss = ((noise_pred - noise) ** 2).mean()
        return loss


# ============================================================
# DDIM Sampler
# ============================================================

class DDIMSampler:
    """
    DDIM deterministic sampler (η=0).

    Given ε_θ(x_t, t), the DDIM update is:
        x̂_0 = (x_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t
        x_{t-1} = √ᾱ_{t-1} · x̂_0 + √(1-ᾱ_{t-1}) · ε_θ
    """

    def __init__(self, model: nn.Module, num_timesteps: int = 1000):
        self.model = model
        self.schedule = DiffusionSchedule(num_timesteps)
        self.T = num_timesteps

    @torch.no_grad()
    def sample(
        self,
        x_T: torch.Tensor,
        num_steps: int = 50,
        t_start_frac: float = 1.0,
        t_end_frac: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM sampling from t_start to t_end.

        Args:
            x_T: initial noisy input [B, C, H, W]
            num_steps: number of DDIM steps (sub-sampled from T total)
            t_start_frac: starting point as fraction of T (1.0 = pure noise)
            t_end_frac: ending point as fraction of T (0.0 = clean image)
        """
        device = x_T.device
        self.schedule.to(device)

        # Build sub-sampled timestep sequence (descending)
        t_start_idx = int(t_start_frac * (self.T - 1))
        t_end_idx = int(t_end_frac * (self.T - 1))
        timesteps = np.linspace(t_start_idx, t_end_idx, num_steps + 1,
                                dtype=int)

        x = x_T.clone()

        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]

            t_batch = torch.full((x.shape[0],), t_cur, device=device,
                                 dtype=torch.long)
            t_norm = t_batch.float() / self.T

            # Predict noise
            eps_pred = self.model(x, t_norm)

            # Current and next alpha_bar
            ab_cur = self.schedule.alpha_bar[t_cur]
            ab_next = self.schedule.alpha_bar[t_next] if t_next >= 0 else \
                torch.tensor(1.0, device=device)

            # DDIM update
            x0_pred = (x - torch.sqrt(1 - ab_cur) * eps_pred) / \
                torch.sqrt(ab_cur)
            x0_pred = x0_pred.clamp(-1, 1)  # clip for stability
            x = torch.sqrt(ab_next) * x0_pred + \
                torch.sqrt(1 - ab_next) * eps_pred

        return x

    @torch.no_grad()
    def get_noise_prediction(self, x: torch.Tensor,
                             t_frac: float) -> torch.Tensor:
        """Get ε_θ(x, t) at a specific point."""
        t_idx = int(t_frac * (self.T - 1))
        t_batch = torch.full((x.shape[0],), t_idx, device=x.device,
                             dtype=torch.long)
        t_norm = t_batch.float() / self.T
        return self.model(x, t_norm)

    @torch.no_grad()
    def get_velocity_equivalent(self, x: torch.Tensor,
                                t_frac: float) -> torch.Tensor:
        """
        Compute the effective 'velocity' of the DDIM ODE at (x, t).

        The DDIM ODE can be written as dx/dt = f(x, t), and this returns
        that f(x, t) for comparison with RF velocity.
        """
        t_idx = int(t_frac * (self.T - 1))
        t_batch = torch.full((x.shape[0],), t_idx, device=x.device,
                             dtype=torch.long)
        t_norm = t_batch.float() / self.T

        eps_pred = self.model(x, t_norm)

        ab = self.schedule.alpha_bar[t_idx]
        # effective velocity = dx/dt for DDIM ODE
        x0_pred = (x - torch.sqrt(1 - ab) * eps_pred) / torch.sqrt(ab)
        # direction pointing from x_t toward x_0 prediction
        velocity = x0_pred - x
        return velocity

    @torch.no_grad()
    def sample_with_trajectory(
        self,
        x_T: torch.Tensor,
        num_steps: int = 50,
        record_every: int = 1,
    ) -> tuple:
        """Sample while recording full trajectory."""
        device = x_T.device
        self.schedule.to(device)

        timesteps = np.linspace(self.T - 1, 0, num_steps + 1, dtype=int)
        x = x_T.clone()
        trajectory = [(1.0, x.clone())]

        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]

            t_batch = torch.full((x.shape[0],), t_cur, device=device,
                                 dtype=torch.long)
            t_norm = t_batch.float() / self.T

            eps_pred = self.model(x, t_norm)

            ab_cur = self.schedule.alpha_bar[t_cur]
            ab_next = self.schedule.alpha_bar[t_next] if t_next >= 0 else \
                torch.tensor(1.0, device=device)

            x0_pred = (x - torch.sqrt(1 - ab_cur) * eps_pred) / \
                torch.sqrt(ab_cur)
            x0_pred = x0_pred.clamp(-1, 1)
            x = torch.sqrt(ab_next) * x0_pred + \
                torch.sqrt(1 - ab_next) * eps_pred

            t_frac = t_next / (self.T - 1)
            if (i + 1) % record_every == 0:
                trajectory.append((t_frac, x.clone()))

        return x, trajectory


# ============================================================
# DC on DDIM — Direction Correction applied to DDIM
# ============================================================

class DCOnDDIMSampler:
    """
    Applies the Direction Correction protocol to DDIM sampling.

    This is the controlled comparison: same protocol as DC-on-RF,
    but using DDIM's noise prediction instead of RF's velocity.

    The correction is: δε = ε_cloud(x_t*, t*) - ε_edge(x_t*, t*)
    Applied as: ε_corrected(x_t, t) = ε_edge(x_t, t) + δε  for t > t*

    Hypothesis: this should work MUCH worse than DC on RF because
    the noise prediction changes rapidly along DDIM's curved trajectory.
    """

    def __init__(self, edge_model: nn.Module, cloud_model: nn.Module,
                 num_timesteps: int = 1000):
        self.edge_sampler = DDIMSampler(edge_model, num_timesteps)
        self.cloud_sampler = DDIMSampler(cloud_model, num_timesteps)
        self.edge_model = edge_model
        self.cloud_model = cloud_model
        self.schedule = DiffusionSchedule(num_timesteps)
        self.T = num_timesteps

    @torch.no_grad()
    def sample_direction_correction(
        self,
        x_T: torch.Tensor,
        t_star: float = 0.5,
        edge_steps_before: int = 10,
        edge_steps_after: int = 10,
        compress_fn=None,
    ) -> dict:
        """
        DC protocol on DDIM.

        t_star here is in [0, 1] where 1=noise, 0=clean (reversed from RF).
        To keep comparable with RF experiments, we map:
            RF t_star=0.7 means "edge has done 70% of the path"
            DDIM t_star=0.7 means "edge samples from T to 0.3*T, then DC"
        """
        device = x_T.device
        self.schedule.to(device)

        # Phase 1: Edge runs DDIM from t=1.0 to t=(1-t_star)
        # (edge has traversed t_star fraction of the full path)
        ddim_t_split = 1.0 - t_star  # fraction of T remaining
        x_split = self.edge_sampler.sample(
            x_T, num_steps=edge_steps_before,
            t_start_frac=1.0, t_end_frac=ddim_t_split,
        )

        # Phase 2: Compute δε at the split point
        t_idx = int(ddim_t_split * (self.T - 1))
        t_batch = torch.full((x_T.shape[0],), t_idx, device=device,
                             dtype=torch.long)
        t_norm = t_batch.float() / self.T

        eps_edge = self.edge_model(x_split, t_norm)
        eps_cloud = self.cloud_model(x_split, t_norm)
        delta_eps = eps_cloud - eps_edge

        if compress_fn is not None:
            delta_eps = compress_fn(delta_eps)

        # Phase 3: Edge continues DDIM from ddim_t_split to 0, adding δε
        timesteps = np.linspace(
            int(ddim_t_split * (self.T - 1)), 0,
            edge_steps_after + 1, dtype=int
        )

        x = x_split.clone()
        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]

            t_b = torch.full((x.shape[0],), t_cur, device=device,
                             dtype=torch.long)
            t_n = t_b.float() / self.T

            eps_pred = self.edge_model(x, t_n)
            # Apply direction correction
            eps_corrected = eps_pred + delta_eps

            ab_cur = self.schedule.alpha_bar[t_cur]
            ab_next = self.schedule.alpha_bar[t_next] if t_next >= 0 else \
                torch.tensor(1.0, device=device)

            x0_pred = (x - torch.sqrt(1 - ab_cur) * eps_corrected) / \
                torch.sqrt(ab_cur)
            x0_pred = x0_pred.clamp(-1, 1)
            x = torch.sqrt(ab_next) * x0_pred + \
                torch.sqrt(1 - ab_next) * eps_corrected

        return {
            'x_final': x,
            'delta_eps': delta_eps,
            'x_split': x_split,
        }

    @torch.no_grad()
    def sample_multi_point_correction(
        self,
        x_T: torch.Tensor,
        query_points: list[float] = [0.3, 0.7],
        total_steps: int = 20,
        compress_fn=None,
    ) -> dict:
        """
        Multi-point DC on DDIM.

        query_points are in RF convention: fraction of path completed.
        We convert to DDIM convention internally.
        """
        device = x_T.device
        self.schedule.to(device)

        query_points = sorted(query_points)
        # Convert to DDIM time fractions (descending)
        ddim_splits = [1.0 - q for q in query_points]  # descending

        # Build boundaries in DDIM time: [1.0, split1, split2, ..., 0.0]
        boundaries = [1.0] + ddim_splits + [0.0]

        x = x_T.clone()
        delta_epsilons = []

        for i in range(len(boundaries) - 1):
            t_start_frac = boundaries[i]
            t_end_frac = boundaries[i + 1]

            seg_steps = max(1, round(
                total_steps * (t_start_frac - t_end_frac)))

            # At each query point (after first segment), compute δε
            if i > 0:
                t_idx = int(t_start_frac * (self.T - 1))
                t_batch = torch.full((x.shape[0],), t_idx, device=device,
                                     dtype=torch.long)
                t_norm = t_batch.float() / self.T

                eps_edge = self.edge_model(x, t_norm)
                eps_cloud = self.cloud_model(x, t_norm)
                delta_eps = eps_cloud - eps_edge
                if compress_fn is not None:
                    delta_eps = compress_fn(delta_eps)
                delta_epsilons.append(delta_eps)
                current_de = delta_eps
            else:
                current_de = None

            # Run DDIM steps with correction
            timesteps = np.linspace(
                int(t_start_frac * (self.T - 1)),
                int(t_end_frac * (self.T - 1)),
                seg_steps + 1, dtype=int
            )

            for j in range(len(timesteps) - 1):
                t_cur = timesteps[j]
                t_next = timesteps[j + 1]

                t_b = torch.full((x.shape[0],), t_cur, device=device,
                                 dtype=torch.long)
                t_n = t_b.float() / self.T

                eps_pred = self.edge_model(x, t_n)
                if current_de is not None:
                    eps_pred = eps_pred + current_de

                ab_cur = self.schedule.alpha_bar[t_cur]
                ab_next = self.schedule.alpha_bar[t_next] if t_next >= 0 \
                    else torch.tensor(1.0, device=device)

                x0_pred = (x - torch.sqrt(1 - ab_cur) * eps_pred) / \
                    torch.sqrt(ab_cur)
                x0_pred = x0_pred.clamp(-1, 1)
                x = torch.sqrt(ab_next) * x0_pred + \
                    torch.sqrt(1 - ab_next) * eps_pred

        return {
            'x_final': x,
            'delta_epsilons': delta_epsilons,
            'num_queries': len(query_points),
        }

    @torch.no_grad()
    def sample_state_relay(
        self,
        x_T: torch.Tensor,
        t_star: float = 0.5,
        edge_steps: int = 10,
        cloud_steps: int = 25,
    ) -> torch.Tensor:
        """State relay baseline on DDIM."""
        ddim_t_split = 1.0 - t_star
        x_split = self.edge_sampler.sample(
            x_T, num_steps=edge_steps,
            t_start_frac=1.0, t_end_frac=ddim_t_split,
        )
        x_final = self.cloud_sampler.sample(
            x_split, num_steps=cloud_steps,
            t_start_frac=ddim_t_split, t_end_frac=0.0,
        )
        return x_final

    @torch.no_grad()
    def sample_edge_only(self, x_T: torch.Tensor,
                         num_steps: int = 20) -> torch.Tensor:
        return self.edge_sampler.sample(x_T, num_steps=num_steps)

    @torch.no_grad()
    def sample_cloud_only(self, x_T: torch.Tensor,
                          num_steps: int = 50) -> torch.Tensor:
        return self.cloud_sampler.sample(x_T, num_steps=num_steps)


# ============================================================
# Trajectory analysis utilities
# ============================================================

@torch.no_grad()
def compute_trajectory_straightness(
    model: nn.Module,
    x_init: torch.Tensor,
    sampler_type: str = 'rf',
    num_steps: int = 100,
    num_timesteps: int = 1000,
) -> dict:
    """
    Compute Trajectory Straightness Score (TSS).

    TSS = ||x_final - x_init||₂ / arc_length
    TSS = 1.0 for perfectly straight trajectories.

    Args:
        model: the generative model
        x_init: initial state (noise for RF/DDIM)
        sampler_type: 'rf' or 'ddim'
        num_steps: integration steps
    """
    device = x_init.device
    B = x_init.shape[0]

    if sampler_type == 'rf':
        from rectified_flow import RectifiedFlowSampler
        sampler = RectifiedFlowSampler(model)
        x_final, traj = sampler.sample_with_trajectory(
            x_init, num_steps=num_steps, record_every=1)
    else:
        sampler = DDIMSampler(model, num_timesteps)
        sampler.schedule.to(device)
        x_final, traj = sampler.sample_with_trajectory(
            x_init, num_steps=num_steps, record_every=1)

    # Chord length: ||x_final - x_init||
    chord = (x_final - x_init).reshape(B, -1).norm(dim=1)

    # Arc length: sum of ||x_{i+1} - x_i|| along trajectory
    arc_length = torch.zeros(B, device=device)
    for i in range(len(traj) - 1):
        _, x_i = traj[i]
        _, x_ip1 = traj[i + 1]
        seg = (x_ip1 - x_i).reshape(B, -1).norm(dim=1)
        arc_length += seg

    tss = chord / arc_length.clamp(min=1e-8)

    return {
        'tss_mean': tss.mean().item(),
        'tss_std': tss.std().item(),
        'tss_per_sample': tss.cpu().numpy(),
        'chord_mean': chord.mean().item(),
        'arc_length_mean': arc_length.mean().item(),
    }


@torch.no_grad()
def compute_delta_consistency(
    edge_model: nn.Module,
    cloud_model: nn.Module,
    x_init: torch.Tensor,
    t_star: float = 0.5,
    sampler_type: str = 'rf',
    num_steps: int = 100,
    num_probe_points: int = 9,
    num_timesteps: int = 1000,
) -> dict:
    """
    Measure how much δv (or δε) changes along the trajectory.

    Computes ||δ(t) - δ(t*)||₂ / ||δ(t*)||₂ at multiple time points.

    For RF: δ = v_cloud - v_edge
    For DDIM: δ = ε_cloud - ε_edge
    """
    device = x_init.device
    B = x_init.shape[0]

    probe_times = np.linspace(0.1, 0.9, num_probe_points)

    if sampler_type == 'rf':
        from rectified_flow import RectifiedFlowSampler
        edge_sampler = RectifiedFlowSampler(edge_model)

        # First, run edge to generate trajectory states at each probe time
        states = {}
        for t_probe in probe_times:
            x_t = edge_sampler.sample(x_init, num_steps=num_steps,
                                      t_start=0.0, t_end=t_probe)
            states[t_probe] = x_t

        # Compute δv at each probe point
        deltas = {}
        for t_probe in probe_times:
            x_t = states[t_probe]
            t_batch = torch.full((B,), t_probe, device=device)
            v_edge = edge_model(x_t, t_batch)
            v_cloud = cloud_model(x_t, t_batch)
            deltas[t_probe] = (v_cloud - v_edge).reshape(B, -1)

    else:
        schedule = DiffusionSchedule(num_timesteps)
        schedule.to(device)
        edge_sampler = DDIMSampler(edge_model, num_timesteps)
        edge_sampler.schedule.to(device)

        states = {}
        for t_probe in probe_times:
            ddim_t = 1.0 - t_probe
            x_t = edge_sampler.sample(x_init, num_steps=num_steps,
                                      t_start_frac=1.0, t_end_frac=ddim_t)
            states[t_probe] = x_t

        deltas = {}
        for t_probe in probe_times:
            x_t = states[t_probe]
            ddim_t = 1.0 - t_probe
            t_idx = int(ddim_t * (num_timesteps - 1))
            t_batch = torch.full((B,), t_idx, device=device,
                                 dtype=torch.long)
            t_norm = t_batch.float() / num_timesteps
            eps_edge = edge_model(x_t, t_norm)
            eps_cloud = cloud_model(x_t, t_norm)
            deltas[t_probe] = (eps_cloud - eps_edge).reshape(B, -1)

    # Find δ at t_star (closest probe point)
    t_star_probe = min(probe_times, key=lambda t: abs(t - t_star))
    delta_star = deltas[t_star_probe]
    delta_star_norm = delta_star.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Compute relative difference at each probe point
    consistency = {}
    for t_probe in probe_times:
        diff = (deltas[t_probe] - delta_star).norm(dim=1)
        relative_diff = diff / delta_star_norm.squeeze()
        consistency[float(t_probe)] = {
            'relative_diff_mean': relative_diff.mean().item(),
            'relative_diff_std': relative_diff.std().item(),
        }

    return {
        't_star': float(t_star_probe),
        'probe_times': [float(t) for t in probe_times],
        'consistency': consistency,
    }


@torch.no_grad()
def compute_delta_spectrum(
    edge_model: nn.Module,
    cloud_model: nn.Module,
    x_init: torch.Tensor,
    t_star: float = 0.5,
    sampler_type: str = 'rf',
    num_steps: int = 50,
    top_k: int = 50,
    num_timesteps: int = 1000,
) -> dict:
    """
    SVD analysis of δv (or δε) to characterize its energy concentration.

    Low-rank δ → energy concentrated in few singular values → compressible.
    """
    device = x_init.device
    B = x_init.shape[0]

    if sampler_type == 'rf':
        from rectified_flow import RectifiedFlowSampler
        edge_sampler = RectifiedFlowSampler(edge_model)
        x_t = edge_sampler.sample(x_init, num_steps=num_steps,
                                  t_start=0.0, t_end=t_star)
        t_batch = torch.full((B,), t_star, device=device)
        v_edge = edge_model(x_t, t_batch)
        v_cloud = cloud_model(x_t, t_batch)
        delta = (v_cloud - v_edge)
    else:
        schedule = DiffusionSchedule(num_timesteps)
        schedule.to(device)
        edge_sampler = DDIMSampler(edge_model, num_timesteps)
        edge_sampler.schedule.to(device)
        ddim_t = 1.0 - t_star
        x_t = edge_sampler.sample(x_init, num_steps=num_steps,
                                  t_start_frac=1.0, t_end_frac=ddim_t)
        t_idx = int(ddim_t * (num_timesteps - 1))
        t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)
        t_norm = t_batch.float() / num_timesteps
        eps_edge = edge_model(x_t, t_norm)
        eps_cloud = cloud_model(x_t, t_norm)
        delta = (eps_cloud - eps_edge)

    # Reshape to [B, C*H*W] and compute SVD on the batch matrix
    delta_2d = delta.reshape(B, -1)  # [B, D]
    # SVD of the batch: treat B as samples, D as features
    U, S, Vh = torch.linalg.svd(delta_2d, full_matrices=False)

    S_np = S.cpu().numpy()
    total_energy = (S_np ** 2).sum()
    cumulative_energy = np.cumsum(S_np ** 2) / total_energy

    # Find k for 90%, 95%, 99% energy
    k_90 = int(np.searchsorted(cumulative_energy, 0.90)) + 1
    k_95 = int(np.searchsorted(cumulative_energy, 0.95)) + 1
    k_99 = int(np.searchsorted(cumulative_energy, 0.99)) + 1

    return {
        'singular_values': S_np[:top_k].tolist(),
        'cumulative_energy': cumulative_energy[:top_k].tolist(),
        'total_energy': float(total_energy),
        'k_for_90pct': k_90,
        'k_for_95pct': k_95,
        'k_for_99pct': k_99,
        'num_samples': B,
        'feature_dim': delta_2d.shape[1],
    }
