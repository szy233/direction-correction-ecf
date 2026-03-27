"""
visualize_2d_why_flow.py — 2D toy visualization: Why DC requires Flow Models.

Generates a paper-ready figure showing:
  (a) RF straight trajectories + DC correction stays valid
  (b) DDIM curved trajectories + DC correction diverges

Target distribution: 8-mode Gaussian mixture on a circle.
Models: tiny 3-layer MLPs (cloud=256-dim, edge=64-dim).

Usage:
    python visualize_2d_why_flow.py              # train + visualize
    python visualize_2d_why_flow.py --skip_train  # load existing models
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection

# ============================================================
# 2D target distribution
# ============================================================

def sample_8gaussians(n, std=0.05):
    """Sample from 8-mode Gaussian mixture on a circle of radius 2."""
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    centers = np.stack([2 * np.cos(angles), 2 * np.sin(angles)], axis=1)
    indices = np.random.randint(0, 8, n)
    samples = centers[indices] + std * np.random.randn(n, 2)
    return torch.tensor(samples, dtype=torch.float32)


# ============================================================
# Tiny MLP for 2D velocity / noise prediction
# ============================================================

class TinyMLP(nn.Module):
    """Small MLP: (x, t) -> v or epsilon, with time embedding."""

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),  # 2D input + 1D time
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, t):
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        inp = torch.cat([x, t.unsqueeze(-1)], dim=-1)
        return self.net(inp)


# ============================================================
# RF training
# ============================================================

def train_rf(model, epochs=3000, batch_size=512, lr=1e-3):
    """Train Rectified Flow: v_θ(x_t, t) ≈ x_1 - x_0."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(epochs):
        x_1 = sample_8gaussians(batch_size)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(batch_size)
        x_t = (1 - t[:, None]) * x_0 + t[:, None] * x_1
        v_target = x_1 - x_0

        v_pred = model(x_t, t)
        loss = ((v_pred - v_target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 500 == 0:
            print(f"  RF epoch {epoch+1}: loss={loss.item():.6f}")
    return model


# ============================================================
# DDIM training
# ============================================================

def cosine_alpha_bar(t, s=0.008):
    """Cosine schedule alpha_bar at continuous t in [0,1]."""
    return torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2


def train_ddim(model, epochs=3000, batch_size=512, lr=1e-3):
    """Train DDIM: ε_θ(x_t, t) ≈ ε."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(epochs):
        x_0 = sample_8gaussians(batch_size)
        eps = torch.randn_like(x_0)
        t = torch.rand(batch_size)

        ab = cosine_alpha_bar(t)[:, None]
        x_t = torch.sqrt(ab) * x_0 + torch.sqrt(1 - ab) * eps

        eps_pred = model(x_t, t)
        loss = ((eps_pred - eps) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 500 == 0:
            print(f"  DDIM epoch {epoch+1}: loss={loss.item():.6f}")
    return model


# ============================================================
# Sampling with trajectory recording
# ============================================================

@torch.no_grad()
def sample_rf_trajectory(model, x_0, steps=100):
    """RF Euler sampling, returns full trajectory [steps+1, B, 2]."""
    dt = 1.0 / steps
    traj = [x_0.clone()]
    x = x_0.clone()
    for i in range(steps):
        t = torch.full((x.shape[0],), i * dt)
        v = model(x, t)
        x = x + v * dt
        traj.append(x.clone())
    return torch.stack(traj)  # [steps+1, B, 2]


@torch.no_grad()
def sample_ddim_trajectory(model, x_T, steps=100):
    """DDIM sampling (eta=0), returns trajectory [steps+1, B, 2].
    Time goes from t≈1 (noise) to t=0 (data)."""
    ts = torch.linspace(0.999, 0.0, steps + 1)
    traj = [x_T.clone()]
    x = x_T.clone()

    for i in range(steps):
        t_cur = ts[i]
        t_next = ts[i + 1]

        t_batch = torch.full((x.shape[0],), t_cur)
        eps_pred = model(x, t_batch)

        ab_cur = cosine_alpha_bar(t_cur)
        ab_next = cosine_alpha_bar(t_next) if t_next > 0 else torch.tensor(1.0)

        x0_pred = (x - torch.sqrt(1 - ab_cur) * eps_pred) / torch.sqrt(ab_cur)
        x = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1 - ab_next) * eps_pred
        traj.append(x.clone())

    return torch.stack(traj)  # [steps+1, B, 2]


# ============================================================
# Direction Correction sampling
# ============================================================

@torch.no_grad()
def sample_rf_dc(edge, cloud, x_0, t_star_step, steps=100):
    """RF with Direction Correction at t_star."""
    dt = 1.0 / steps
    traj = [x_0.clone()]
    x = x_0.clone()
    delta_v = None

    for i in range(steps):
        t_val = i * dt
        t = torch.full((x.shape[0],), t_val)
        v_edge = edge(x, t)

        if i == t_star_step:
            v_cloud = cloud(x, t)
            delta_v = v_cloud - v_edge

        if delta_v is not None:
            v_edge = v_edge + delta_v

        x = x + v_edge * dt
        traj.append(x.clone())

    return torch.stack(traj)


@torch.no_grad()
def sample_ddim_dc(edge, cloud, x_T, t_star_step, steps=100):
    """DDIM with Direction Correction (δε) at t_star."""
    ts = torch.linspace(0.999, 0.0, steps + 1)
    traj = [x_T.clone()]
    x = x_T.clone()
    delta_eps = None

    for i in range(steps):
        t_cur = ts[i]
        t_next = ts[i + 1]

        t_batch = torch.full((x.shape[0],), t_cur)
        eps_edge = edge(x, t_batch)

        if i == t_star_step:
            eps_cloud = cloud(x, t_batch)
            delta_eps = eps_cloud - eps_edge

        if delta_eps is not None:
            eps_edge = eps_edge + delta_eps

        ab_cur = cosine_alpha_bar(t_cur)
        ab_next = cosine_alpha_bar(t_next) if t_next > 0 else torch.tensor(1.0)

        x0_pred = (x - torch.sqrt(1 - ab_cur) * eps_edge) / torch.sqrt(ab_cur)
        x = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1 - ab_next) * eps_edge
        traj.append(x.clone())

    return torch.stack(traj)


# ============================================================
# Visualization
# ============================================================

def plot_trajectories(ax, traj, color='blue', alpha=0.15, lw=0.6, label=None):
    """Plot trajectories as colored lines with time gradient."""
    n_steps, n_traj, _ = traj.shape
    for j in range(n_traj):
        points = traj[:, j, :].numpy()
        # Color gradient from light to dark
        segments = np.array([[points[i], points[i+1]]
                             for i in range(len(points)-1)])
        t_vals = np.linspace(0.3, 1.0, len(segments))
        colors = plt.cm.colors.to_rgba_array(
            [matplotlib.colors.to_rgba(color, a * alpha) for a in t_vals])
        lc = LineCollection(segments, colors=colors, linewidths=lw)
        ax.add_collection(lc)
    # dummy for legend
    if label:
        ax.plot([], [], color=color, alpha=0.6, lw=1.5, label=label)


def plot_correction_arrows(ax, traj, t_star_step, delta, color='red'):
    """Draw the correction vector δ at the split point."""
    for j in range(min(traj.shape[1], 8)):
        x_at_tstar = traj[t_star_step, j, :].detach().numpy()
        d = delta[j].detach().numpy()
        ax.annotate('', xy=(x_at_tstar[0] + d[0] * 0.5,
                            x_at_tstar[1] + d[1] * 0.5),
                     xytext=(x_at_tstar[0], x_at_tstar[1]),
                     arrowprops=dict(arrowstyle='->', color=color,
                                     lw=1.5, mutation_scale=10))


def plot_trajectory_with_dots(ax, traj, color, alpha=0.6, lw=1.5, label=None,
                              dot_interval=10, dot_size=15, marker='o'):
    """Plot trajectories with intermediate dots to highlight curvature."""
    n_steps, n_traj, _ = traj.shape
    for j in range(n_traj):
        points = traj[:, j, :].numpy()
        ax.plot(points[:, 0], points[:, 1], color=color, alpha=alpha, lw=lw,
                zorder=3)
        # Intermediate dots at regular intervals
        for s in range(0, n_steps, dot_interval):
            ax.scatter(points[s, 0], points[s, 1], c=color, s=dot_size,
                       zorder=4, alpha=alpha * 0.8, marker=marker,
                       edgecolors='white', linewidths=0.3)
    if label:
        ax.plot([], [], color=color, alpha=alpha, lw=2, label=label, marker=marker,
                markersize=4)


def draw_figure(rf_cloud, rf_edge, ddim_cloud, ddim_edge, save_path, steps=100):
    """Generate the main comparison figure: 1x2 layout, RF vs DDIM on same plot."""
    torch.manual_seed(7)
    n_traj = 8
    x_0 = torch.randn(n_traj, 2)  # shared initial noise for both RF and DDIM

    t_star_step = int(steps * 0.5)  # t* = 0.5

    # --- Generate trajectories (same x_0 for both) ---
    rf_cloud_traj = sample_rf_trajectory(rf_cloud, x_0, steps)
    rf_edge_traj = sample_rf_trajectory(rf_edge, x_0, steps)
    rf_dc_traj = sample_rf_dc(rf_edge, rf_cloud, x_0, t_star_step, steps)

    ddim_cloud_traj = sample_ddim_trajectory(ddim_cloud, x_0, steps)
    ddim_edge_traj = sample_ddim_trajectory(ddim_edge, x_0, steps)
    ddim_dc_traj = sample_ddim_dc(ddim_edge, ddim_cloud, x_0, t_star_step, steps)

    # --- Target distribution for background ---
    x_target = sample_8gaussians(2000).numpy()

    # --- Figure: 1x3 layout ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    lim = 3.8

    for ax in axes:
        ax.scatter(x_target[:, 0], x_target[:, 1],
                   c='#e0e0e0', s=3, alpha=0.4, zorder=0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=9)

    # ---- Panel (a): Trajectory comparison ----
    ax = axes[0]
    ax.set_title('(a) Sampling Trajectories (same noise)',
                 fontsize=13, fontweight='bold', pad=10)

    plot_trajectory_with_dots(ax, rf_cloud_traj, color='#1565C0', alpha=0.6,
                              lw=2.0, label='RF (straight)', dot_interval=15,
                              dot_size=18, marker='o')
    plot_trajectory_with_dots(ax, ddim_cloud_traj, color='#C62828', alpha=0.6,
                              lw=2.0, label='DDIM (curved)', dot_interval=15,
                              dot_size=18, marker='s')
    ax.scatter(x_0[:, 0].numpy(), x_0[:, 1].numpy(),
               c='black', s=60, zorder=7, marker='*', linewidths=1.0,
               label='Shared start')
    ax.scatter(rf_cloud_traj[-1, :, 0].numpy(),
               rf_cloud_traj[-1, :, 1].numpy(),
               c='#1565C0', s=50, zorder=6, edgecolors='black', linewidths=1.0,
               marker='o')
    ax.scatter(ddim_cloud_traj[-1, :, 0].numpy(),
               ddim_cloud_traj[-1, :, 1].numpy(),
               c='#C62828', s=50, zorder=6, edgecolors='black', linewidths=1.0,
               marker='s')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95,
              edgecolor='#cccccc')
    ax.text(0.97, 0.03, 'Same noise, different paths\nRF: straight  |  DDIM: curved',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, style='italic', color='#333333',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffffcc',
                      alpha=0.9, edgecolor='#999999'))

    # ---- Panel (b): RF + DC ----
    ax = axes[1]
    ax.set_title('(b) RF + Direction Correction',
                 fontsize=13, fontweight='bold', pad=10)

    plot_trajectory_with_dots(ax, rf_edge_traj[:t_star_step+1], color='#FF9800',
                              alpha=0.5, lw=1.5, label='Edge (before t*)',
                              dot_interval=10, dot_size=12)
    plot_trajectory_with_dots(ax, rf_dc_traj[t_star_step:], color='#2E7D32',
                              alpha=0.7, lw=2.5, label='Edge+DC (corrected)',
                              dot_interval=10, dot_size=20, marker='o')
    ax.scatter(rf_dc_traj[-1, :, 0].numpy(),
               rf_dc_traj[-1, :, 1].numpy(),
               c='#2E7D32', s=60, zorder=6, edgecolors='black', linewidths=1.0,
               marker='o')
    ax.scatter(x_0[:, 0].numpy(), x_0[:, 1].numpy(),
               c='black', s=50, zorder=7, marker='*', linewidths=1.0)
    ax.scatter(rf_edge_traj[t_star_step, :, 0].numpy(),
               rf_edge_traj[t_star_step, :, 1].numpy(),
               c='#E91E63', s=50, zorder=6, marker='D', edgecolors='white',
               linewidths=0.5)
    ax.plot([], [], color='#E91E63', marker='D', linestyle='none', markersize=6,
            label='t* = 0.5 (query)')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95,
              edgecolor='#cccccc')
    ax.text(0.97, 0.03, 'Straight trajectory\n→ δv stays valid, reaches target',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, style='italic', color='#2E7D32',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f5e9',
                      alpha=0.9, edgecolor='#2E7D32'))

    # ---- Panel (c): DDIM + DC ----
    ax = axes[2]
    ax.set_title('(c) DDIM + Direction Correction',
                 fontsize=13, fontweight='bold', pad=10)

    plot_trajectory_with_dots(ax, ddim_edge_traj[:t_star_step+1], color='#9C27B0',
                              alpha=0.5, lw=1.5, label='Edge (before t*)',
                              dot_interval=10, dot_size=12)
    plot_trajectory_with_dots(ax, ddim_dc_traj[t_star_step:], color='#C62828',
                              alpha=0.7, lw=2.5, label='Edge+DC (diverges)',
                              dot_interval=10, dot_size=20, marker='s')
    ax.scatter(ddim_dc_traj[-1, :, 0].numpy(),
               ddim_dc_traj[-1, :, 1].numpy(),
               c='#C62828', s=60, zorder=6, edgecolors='black', linewidths=1.0,
               marker='s')
    ax.scatter(x_0[:, 0].numpy(), x_0[:, 1].numpy(),
               c='black', s=50, zorder=7, marker='*', linewidths=1.0)
    ax.scatter(ddim_edge_traj[t_star_step, :, 0].numpy(),
               ddim_edge_traj[t_star_step, :, 1].numpy(),
               c='#E91E63', s=50, zorder=6, marker='D', edgecolors='white',
               linewidths=0.5)
    ax.plot([], [], color='#E91E63', marker='D', linestyle='none', markersize=6,
            label='t* = 0.5 (query)')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95,
              edgecolor='#cccccc')
    ax.text(0.97, 0.03, 'Curved trajectory\n→ δε becomes invalid, drifts away',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, style='italic', color='#C62828',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffebee',
                      alpha=0.9, edgecolor='#C62828'))

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved figure to {save_path}")
    plt.close()


def draw_delta_validity_figure(rf_cloud, rf_edge, ddim_cloud, ddim_edge,
                                save_path, steps=100):
    """
    Supplementary figure: δ validity over time.

    Shows ||δ(t) - δ(t*)||/||δ(t*)|| for RF vs DDIM — a single clean plot
    demonstrating RF's correction stays valid while DDIM's diverges.
    """
    torch.manual_seed(42)
    n = 256
    x_0 = torch.randn(n, 2)

    t_star_step = int(steps * 0.5)
    probe_steps = list(range(0, steps, max(1, steps // 20)))

    results = {}

    for name, cloud, edge, sample_fn, is_ddim in [
        ('Rectified Flow', rf_cloud, rf_edge,
         lambda m, x: sample_rf_trajectory(m, x, steps), False),
        ('DDIM', ddim_cloud, ddim_edge,
         lambda m, x: sample_ddim_trajectory(m, x, steps), True),
    ]:
        edge_traj = sample_fn(edge, x_0)

        # Compute δ at t*
        x_tstar = edge_traj[t_star_step]
        if is_ddim:
            # DDIM time: step 0 = t=1.0 (noise), step N = t=0.0 (data)
            t_b = torch.full((n,), 1.0 - t_star_step / steps)
        else:
            t_b = torch.full((n,), t_star_step / steps)
        delta_star = cloud(x_tstar, t_b) - edge(x_tstar, t_b)
        delta_star_norm = delta_star.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # Compute δ at each probe step
        diffs = []
        t_fracs = []
        for s in probe_steps:
            x_s = edge_traj[s]
            if is_ddim:
                t_b = torch.full((n,), 1.0 - s / steps)
            else:
                t_b = torch.full((n,), s / steps)
            delta_s = cloud(x_s, t_b) - edge(x_s, t_b)
            rel_diff = (delta_s - delta_star).norm(dim=1) / \
                delta_star_norm.squeeze()
            diffs.append(rel_diff.mean().item())
            t_fracs.append(s / steps)

        results[name] = (t_fracs, diffs)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

    t_rf, d_rf = results['Rectified Flow']
    t_ddim, d_ddim = results['DDIM']

    ax.plot(t_rf, d_rf, 'o-', color='#4CAF50', lw=2, markersize=4,
            label='Rectified Flow δv', zorder=5)
    ax.plot(t_ddim, d_ddim, 's-', color='#F44336', lw=2, markersize=4,
            label='DDIM δε', zorder=5)

    ax.axvline(x=0.5, color='#E91E63', ls='--', lw=1, alpha=0.7,
               label='t* = 0.5')
    ax.axhspan(0, 0.5, alpha=0.05, color='green',
               label='Effective correction zone')

    ax.set_xlabel('Time t (fraction of trajectory)', fontsize=12)
    ax.set_ylabel('||δ(t) - δ(t*)|| / ||δ(t*)||', fontsize=12)
    ax.set_title('Correction Vector Validity Along Trajectory',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved figure to {save_path}")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--output_dir', type=str, default='results/viz_2d')
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, 'models_2d.pt')

    # Cloud: 256-dim hidden, Edge: 64-dim hidden
    rf_cloud = TinyMLP(hidden_dim=256)
    rf_edge = TinyMLP(hidden_dim=64)
    ddim_cloud = TinyMLP(hidden_dim=256)
    ddim_edge = TinyMLP(hidden_dim=64)

    if not args.skip_train:
        print("=== Training RF Cloud ===")
        train_rf(rf_cloud, epochs=args.epochs)
        print("=== Training RF Edge ===")
        train_rf(rf_edge, epochs=args.epochs)
        print("=== Training DDIM Cloud ===")
        train_ddim(ddim_cloud, epochs=args.epochs)
        print("=== Training DDIM Edge ===")
        train_ddim(ddim_edge, epochs=args.epochs)

        torch.save({
            'rf_cloud': rf_cloud.state_dict(),
            'rf_edge': rf_edge.state_dict(),
            'ddim_cloud': ddim_cloud.state_dict(),
            'ddim_edge': ddim_edge.state_dict(),
        }, ckpt_path)
        print(f"Models saved to {ckpt_path}")
    else:
        ckpt = torch.load(ckpt_path, weights_only=True)
        rf_cloud.load_state_dict(ckpt['rf_cloud'])
        rf_edge.load_state_dict(ckpt['rf_edge'])
        ddim_cloud.load_state_dict(ckpt['ddim_cloud'])
        ddim_edge.load_state_dict(ckpt['ddim_edge'])
        print("Loaded existing models.")

    rf_cloud.eval()
    rf_edge.eval()
    ddim_cloud.eval()
    ddim_edge.eval()

    # Figure 1: Main trajectory comparison (2x2)
    print("\n=== Generating main trajectory figure ===")
    draw_figure(rf_cloud, rf_edge, ddim_cloud, ddim_edge,
                os.path.join(args.output_dir, 'why_flow_2d.png'),
                steps=args.steps)

    # Figure 2: δ validity curve
    print("=== Generating δ validity figure ===")
    draw_delta_validity_figure(
        rf_cloud, rf_edge, ddim_cloud, ddim_edge,
        os.path.join(args.output_dir, 'delta_validity_2d.png'),
        steps=args.steps)

    # Figure 3: Endpoint distribution comparison (1x4)
    print("=== Generating endpoint comparison figure ===")
    draw_endpoint_figure(
        rf_cloud, rf_edge, ddim_cloud, ddim_edge,
        os.path.join(args.output_dir, 'endpoints_2d.png'),
        steps=args.steps)

    print("\nDone! Figures saved to:", args.output_dir)


def draw_endpoint_figure(rf_cloud, rf_edge, ddim_cloud, ddim_edge,
                          save_path, steps=100):
    """
    Compare final sample distributions: 4 panels showing where
    samples land for Cloud, Edge, DC-RF, DC-DDIM.
    """
    torch.manual_seed(0)
    n = 512
    x_0 = torch.randn(n, 2)
    t_star_step = int(steps * 0.5)

    x_target = sample_8gaussians(2000).numpy()

    # Generate endpoints
    rf_cloud_end = sample_rf_trajectory(rf_cloud, x_0, steps)[-1].numpy()
    rf_edge_end = sample_rf_trajectory(rf_edge, x_0, steps)[-1].numpy()
    rf_dc_end = sample_rf_dc(rf_edge, rf_cloud, x_0, t_star_step, steps)[-1].numpy()

    ddim_cloud_end = sample_ddim_trajectory(ddim_cloud, x_0, steps)[-1].numpy()
    ddim_edge_end = sample_ddim_trajectory(ddim_edge, x_0, steps)[-1].numpy()
    ddim_dc_end = sample_ddim_dc(ddim_edge, ddim_cloud, x_0, t_star_step, steps)[-1].numpy()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    lim = 3.5

    configs = [
        ('RF: Edge Only', rf_edge_end, '#FF9800'),
        ('RF: Edge + DC', rf_dc_end, '#4CAF50'),
        ('DDIM: Edge Only', ddim_edge_end, '#FF9800'),
        ('DDIM: Edge + DC', ddim_dc_end, '#F44336'),
    ]

    for ax, (title, endpoints, color) in zip(axes, configs):
        ax.scatter(x_target[:, 0], x_target[:, 1],
                   c='#e0e0e0', s=3, alpha=0.4, zorder=0, label='Target')
        ax.scatter(endpoints[:, 0], endpoints[:, 1],
                   c=color, s=8, alpha=0.5, zorder=5, label='Generated')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.tick_params(labelsize=8)

    # Add quality annotations
    axes[1].text(0.97, 0.03, 'DC improves\nmode coverage',
                 transform=axes[1].transAxes, ha='right', va='bottom',
                 fontsize=9, color='#4CAF50', style='italic',
                 bbox=dict(boxstyle='round', facecolor='white',
                           alpha=0.8, edgecolor='#4CAF50'))
    axes[3].text(0.97, 0.03, 'DC causes\nscatter / divergence',
                 transform=axes[3].transAxes, ha='right', va='bottom',
                 fontsize=9, color='#F44336', style='italic',
                 bbox=dict(boxstyle='round', facecolor='white',
                           alpha=0.8, edgecolor='#F44336'))

    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved figure to {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
