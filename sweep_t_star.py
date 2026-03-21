"""
sweep_t_star.py — Experiment 2: Sweep the split point t* to find the optimal partition.

Produces the U-shaped FID vs t* curve that demonstrates the trade-off between
edge error accumulation and correction decay.

Usage:
    python sweep_t_star.py --cloud_ckpt checkpoints/cloud_best.pt \
                           --edge_ckpt checkpoints/edge_best.pt
"""

import os
import argparse
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import build_cloud_model, build_edge_model
from rectified_flow import DirectionCorrectionSampler
from metrics import save_samples_to_dir, compute_fid, prepare_cifar10_reference


def main():
    parser = argparse.ArgumentParser(description="Sweep t* for optimal split point")
    parser.add_argument('--cloud_ckpt', type=str, required=True)
    parser.add_argument('--edge_ckpt', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=5000,
                        help="Samples per t* (lower for faster sweep)")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--total_steps', type=int, default=20,
                        help="Total Euler steps (split between before/after t*)")
    parser.add_argument('--output_dir', type=str, default='results/sweep_t_star/')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    print("Loading models...")
    cloud_model = build_cloud_model()
    cloud_ckpt = torch.load(args.cloud_ckpt, map_location=device)
    cloud_model.load_state_dict(cloud_ckpt['model_state_dict'])
    cloud_model = cloud_model.to(device).eval()

    edge_model = build_edge_model()
    edge_ckpt = torch.load(args.edge_ckpt, map_location=device)
    edge_model.load_state_dict(edge_ckpt['model_state_dict'])
    edge_model = edge_model.to(device).eval()

    collab = DirectionCorrectionSampler(edge_model, cloud_model)
    shape = (3, 32, 32)

    # Prepare reference
    ref_dir = prepare_cifar10_reference()

    # Sweep t*
    t_star_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results_dc = {}      # direction correction FID
    results_relay = {}   # state relay FID (for comparison)
    results_alpha = {}   # mean α values

    for t_star in t_star_values:
        print(f"\n{'='*50}")
        print(f"t* = {t_star:.1f}")
        print(f"{'='*50}")

        # Compute steps before/after proportionally
        steps_before = max(1, int(args.total_steps * t_star))
        steps_after = max(1, args.total_steps - steps_before)

        # Direction Correction
        torch.manual_seed(42)
        alpha_vals = []
        all_samples = []
        remaining = args.num_samples

        while remaining > 0:
            bs = min(args.batch_size, remaining)
            x_0 = torch.randn(bs, *shape, device=device)
            result = collab.sample_direction_correction(
                x_0, t_star=t_star,
                edge_steps_before=steps_before,
                edge_steps_after=steps_after,
            )
            all_samples.append(result['x_final'].cpu())
            alpha_vals.extend(result['alpha_values'])
            remaining -= bs

        samples_dc = torch.cat(all_samples, dim=0)[:args.num_samples]
        dc_dir = save_samples_to_dir(samples_dc, f"{args.output_dir}/dc_t{t_star:.1f}/")
        fid_dc = compute_fid(dc_dir, ref_dir, device=device)
        results_dc[t_star] = fid_dc
        results_alpha[t_star] = float(np.mean(alpha_vals))

        # State Relay (baseline)
        torch.manual_seed(42)
        all_samples_relay = []
        remaining = args.num_samples

        while remaining > 0:
            bs = min(args.batch_size, remaining)
            x_0 = torch.randn(bs, *shape, device=device)
            x_final = collab.sample_state_relay(
                x_0, t_star=t_star,
                edge_steps=steps_before,
                cloud_steps=max(10, int(50 * (1 - t_star))),
            )
            all_samples_relay.append(x_final.cpu())
            remaining -= bs

        samples_relay = torch.cat(all_samples_relay, dim=0)[:args.num_samples]
        relay_dir = save_samples_to_dir(samples_relay, f"{args.output_dir}/relay_t{t_star:.1f}/")
        fid_relay = compute_fid(relay_dir, ref_dir, device=device)
        results_relay[t_star] = fid_relay

        print(f"  Direction Correction FID: {fid_dc:.2f}")
        print(f"  State Relay FID:          {fid_relay:.2f}")
        print(f"  Mean α(t):                {results_alpha[t_star]:.3f}")

    # ---- Plot results ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: FID vs t*
    ts = sorted(results_dc.keys())
    fids_dc = [results_dc[t] for t in ts]
    fids_relay = [results_relay[t] for t in ts]

    ax1.plot(ts, fids_dc, 'o-', color='#2196F3', linewidth=2, markersize=8,
             label='Direction Correction (ours)')
    ax1.plot(ts, fids_relay, 's--', color='#FF9800', linewidth=2, markersize=8,
             label='State Relay (baseline)')
    ax1.set_xlabel('Split Point t*', fontsize=13)
    ax1.set_ylabel('FID ↓', fontsize=13)
    ax1.set_title('FID vs Split Point t*', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Mark optimal t*
    best_t = ts[np.argmin(fids_dc)]
    ax1.axvline(x=best_t, color='#2196F3', linestyle=':', alpha=0.5)
    ax1.annotate(f'Optimal t*={best_t:.1f}',
                 xy=(best_t, min(fids_dc)), fontsize=10,
                 xytext=(best_t + 0.1, min(fids_dc) + 5),
                 arrowprops=dict(arrowstyle='->', color='gray'))

    # Plot 2: Mean α(t) vs t*
    alphas = [results_alpha[t] for t in ts]
    ax2.bar(ts, alphas, width=0.06, color='#4CAF50', alpha=0.8)
    ax2.set_xlabel('Split Point t*', fontsize=13)
    ax2.set_ylabel('Mean α(t)', fontsize=13)
    ax2.set_title('Adaptive Gate Value vs Split Point', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, 'sweep_t_star.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {fig_path}")

    # Save raw results
    all_results = {
        't_star_values': ts,
        'fid_direction_correction': {str(k): v for k, v in results_dc.items()},
        'fid_state_relay': {str(k): v for k, v in results_relay.items()},
        'mean_alpha': {str(k): v for k, v in results_alpha.items()},
        'optimal_t_star': best_t,
    }
    results_path = os.path.join(args.output_dir, 'sweep_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Summary table
    print("\n" + "=" * 55)
    print(f"{'t*':>5s} {'DC FID':>10s} {'Relay FID':>10s} {'α mean':>8s}")
    print("-" * 55)
    for t in ts:
        marker = " ← best" if t == best_t else ""
        print(f"{t:5.1f} {results_dc[t]:10.2f} {results_relay[t]:10.2f} "
              f"{results_alpha[t]:8.3f}{marker}")


if __name__ == "__main__":
    main()
