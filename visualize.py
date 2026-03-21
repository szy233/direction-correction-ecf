"""
visualize.py — Trajectory visualization and qualitative analysis.

Produces:
  - Figure 2: ODE trajectory comparison (cloud vs edge vs corrected)
  - δv sparsity heatmap
  - α(t) decay curve
  - Generated sample grids
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from models import build_cloud_model, build_edge_model
from rectified_flow import RectifiedFlowSampler, DirectionCorrectionSampler


def visualize_trajectories(
    cloud_model, edge_model, device, output_dir='results/viz/', t_star=0.5
):
    """
    Visualize the ODE trajectories of cloud, edge, and corrected paths.
    Uses PCA to project high-dimensional trajectories to 2D.
    """
    os.makedirs(output_dir, exist_ok=True)

    cloud_sampler = RectifiedFlowSampler(cloud_model)
    edge_sampler = RectifiedFlowSampler(edge_model)
    collab = DirectionCorrectionSampler(edge_model, cloud_model)

    # Generate one sample trajectory
    torch.manual_seed(0)
    x_0 = torch.randn(1, 3, 32, 32, device=device)

    # Cloud trajectory
    _, cloud_traj = cloud_sampler.sample_with_trajectory(
        x_0.clone(), num_steps=50, record_every=1
    )

    # Edge trajectory (no correction)
    _, edge_traj = edge_sampler.sample_with_trajectory(
        x_0.clone(), num_steps=20, record_every=1
    )

    # Edge trajectory with direction correction
    # Manual implementation to record trajectory
    edge_model.eval()
    cloud_model.eval()

    steps_before = 10
    steps_after = 10
    dt_before = t_star / steps_before
    dt_after = (1.0 - t_star) / steps_after

    x = x_0.clone()
    corrected_traj = [(0.0, x.clone())]

    # Phase 1: edge alone to t*
    t = 0.0
    for _ in range(steps_before):
        t_batch = torch.full((1,), t, device=device)
        v = edge_model(x, t_batch)
        x = x + v * dt_before
        t += dt_before
        corrected_traj.append((t, x.clone()))

    # Phase 2: compute δv
    x_t_star = x.clone()
    t_batch = torch.full((1,), t_star, device=device)
    v_edge_star = edge_model(x_t_star, t_batch)
    v_cloud_star = cloud_model(x_t_star, t_batch)
    delta_v = v_cloud_star - v_edge_star

    # Phase 3: edge with correction
    v_ref = v_edge_star.clone()
    for _ in range(steps_after):
        t_batch = torch.full((1,), t, device=device)
        v_current = edge_model(x, t_batch)

        # Adaptive α
        cos_sim = torch.nn.functional.cosine_similarity(
            v_current.reshape(1, -1), v_ref.reshape(1, -1)
        ).clamp(min=0)
        v_corrected = v_current + cos_sim[:, None, None, None] * delta_v
        x = x + v_corrected * dt_after
        t += dt_after
        corrected_traj.append((t, x.clone()))

    # Project trajectories to 2D via PCA
    all_points = []
    for _, x_t in cloud_traj:
        all_points.append(x_t.cpu().reshape(-1).numpy())
    for _, x_t in edge_traj:
        all_points.append(x_t.cpu().reshape(-1).numpy())
    for _, x_t in corrected_traj:
        all_points.append(x_t.cpu().reshape(-1).numpy())

    all_points = np.stack(all_points)
    # Center and PCA
    mean = all_points.mean(axis=0)
    centered = all_points - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ Vt[:2].T

    n_cloud = len(cloud_traj)
    n_edge = len(edge_traj)
    n_corr = len(corrected_traj)

    cloud_2d = projected[:n_cloud]
    edge_2d = projected[n_cloud:n_cloud + n_edge]
    corr_2d = projected[n_cloud + n_edge:]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(cloud_2d[:, 0], cloud_2d[:, 1], '-o', color='#4CAF50', markersize=4,
            linewidth=2, label='Cloud (reference)', alpha=0.8)
    ax.plot(edge_2d[:, 0], edge_2d[:, 1], '-s', color='#F44336', markersize=4,
            linewidth=2, label='Edge only', alpha=0.8)
    ax.plot(corr_2d[:, 0], corr_2d[:, 1], '-^', color='#2196F3', markersize=4,
            linewidth=2, label='Direction Correction (ours)', alpha=0.8)

    # Mark t* point
    t_star_idx = steps_before  # index in corrected trajectory
    ax.plot(corr_2d[t_star_idx, 0], corr_2d[t_star_idx, 1], '*',
            color='gold', markersize=20, zorder=5, markeredgecolor='black',
            markeredgewidth=1, label=f't* = {t_star}')

    # Mark start and end
    ax.plot(cloud_2d[0, 0], cloud_2d[0, 1], 'D', color='black', markersize=10,
            zorder=5, label='x₀ (noise)')
    ax.plot(cloud_2d[-1, 0], cloud_2d[-1, 1], 'P', color='black', markersize=10,
            zorder=5, label='x₁ (image)')

    ax.set_xlabel('PC 1', fontsize=13)
    ax.set_ylabel('PC 2', fontsize=13)
    ax.set_title('ODE Trajectory Comparison (PCA projection)', fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(output_dir, 'trajectory_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Trajectory plot saved to {fig_path}")


def visualize_delta_v_sparsity(cloud_model, edge_model, device, output_dir='results/viz/'):
    """Visualize the sparsity pattern of δv."""
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(42)
    x = torch.randn(8, 3, 32, 32, device=device)

    # Run edge to t*=0.5
    edge_sampler = RectifiedFlowSampler(edge_model)
    x_t = edge_sampler.sample(x, num_steps=10, t_start=0.0, t_end=0.5)

    t_batch = torch.full((8,), 0.5, device=device)
    v_edge = edge_model(x_t, t_batch)
    v_cloud = cloud_model(x_t, t_batch)
    delta_v = (v_cloud - v_edge).cpu()

    # Plot δv magnitude heatmap (average over batch and channels)
    dv_mag = delta_v.abs().mean(dim=(0, 1)).numpy()  # [32, 32]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Heatmap
    im = axes[0].imshow(dv_mag, cmap='hot', interpolation='nearest')
    axes[0].set_title('|δv| spatial heatmap', fontsize=12)
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    # Histogram
    dv_flat = delta_v.reshape(-1).numpy()
    axes[1].hist(dv_flat, bins=100, density=True, color='#2196F3', alpha=0.7)
    axes[1].set_title('δv value distribution', fontsize=12)
    axes[1].set_xlabel('δv value')
    axes[1].set_ylabel('Density')

    # Sorted magnitude (shows natural sparsity)
    sorted_mag = np.sort(np.abs(dv_flat))[::-1]
    x_axis = np.arange(len(sorted_mag)) / len(sorted_mag) * 100
    axes[2].plot(x_axis, sorted_mag, color='#4CAF50', linewidth=1.5)
    axes[2].set_title('Sorted |δv| (sparsity profile)', fontsize=12)
    axes[2].set_xlabel('Percentile (%)')
    axes[2].set_ylabel('|δv|')
    axes[2].axvline(x=10, color='red', linestyle='--', alpha=0.5, label='top 10%')
    axes[2].legend()

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'delta_v_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"δv analysis saved to {fig_path}")


def visualize_sample_grid(samples_dict, output_dir='results/viz/'):
    """
    Side-by-side comparison grid of generated samples from different methods.

    Args:
        samples_dict: {method_name: tensor [N, 3, 32, 32]}
    """
    os.makedirs(output_dir, exist_ok=True)

    n_methods = len(samples_dict)
    n_show = 8  # samples per method

    fig, axes = plt.subplots(n_methods, 1, figsize=(16, 3 * n_methods))
    if n_methods == 1:
        axes = [axes]

    for idx, (name, samples) in enumerate(samples_dict.items()):
        # Rescale to [0, 1]
        imgs = (samples[:n_show].clamp(-1, 1) + 1) / 2
        grid = make_grid(imgs, nrow=n_show, padding=2)
        axes[idx].imshow(grid.permute(1, 2, 0).numpy())
        axes[idx].set_title(name, fontsize=13)
        axes[idx].axis('off')

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'sample_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sample grid saved to {fig_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud_ckpt', type=str, required=True)
    parser.add_argument('--edge_ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/viz/')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    cloud_model = build_cloud_model()
    cloud_ckpt = torch.load(args.cloud_ckpt, map_location=device)
    cloud_model.load_state_dict(cloud_ckpt['model_state_dict'])
    cloud_model = cloud_model.to(device).eval()

    edge_model = build_edge_model()
    edge_ckpt = torch.load(args.edge_ckpt, map_location=device)
    edge_model.load_state_dict(edge_ckpt['model_state_dict'])
    edge_model = edge_model.to(device).eval()

    visualize_trajectories(cloud_model, edge_model, device, args.output_dir)
    visualize_delta_v_sparsity(cloud_model, edge_model, device, args.output_dir)
