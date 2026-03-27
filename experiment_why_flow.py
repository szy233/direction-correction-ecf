"""
experiment_why_flow.py — Controlled experiments: Why Direction Correction requires Flow Models.

Experiments:
  Exp6: DC on DDIM vs DC on RF (same architecture, same protocol, different framework)
  Exp7: Trajectory Straightness Score (TSS) comparison
  Exp8: δv / δε temporal consistency along trajectory
  Exp9: δv / δε singular value spectrum (compressibility analysis)

Usage:
    # Step 0: Train DDIM models (same architecture as RF)
    python experiment_why_flow.py --mode train_ddim \
        --epochs 100 --save_dir checkpoints/

    # Step 1: Run all comparison experiments
    python experiment_why_flow.py --mode exp6 \
        --rf_cloud_ckpt checkpoints/cloud_best.pt \
        --rf_edge_ckpt checkpoints/edge_best.pt \
        --ddim_cloud_ckpt checkpoints/ddim_cloud_best.pt \
        --ddim_edge_ckpt checkpoints/ddim_edge_best.pt

    python experiment_why_flow.py --mode exp7 ...
    python experiment_why_flow.py --mode exp8 ...
    python experiment_why_flow.py --mode exp9 ...
    python experiment_why_flow.py --mode all ...
"""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

from models import build_cloud_model, build_edge_model, count_params
from rectified_flow import DirectionCorrectionSampler, RectifiedFlowSampler
from ddim import (DDIMTrainer, DDIMSampler, DCOnDDIMSampler,
                  compute_trajectory_straightness,
                  compute_delta_consistency, compute_delta_spectrum)
from compression import make_compress_fn, compute_compression_ratio
from metrics import compute_fid, save_samples_to_dir, prepare_cifar10_reference


# ============================================================
# Helpers
# ============================================================

def load_model(model_fn, ckpt_path, device):
    model = model_fn()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    print(f"  Loaded {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    return model


def generate_samples(sampler_fn, num_samples, batch_size, shape, device):
    all_samples = []
    remaining = num_samples
    pbar = tqdm(total=num_samples, desc="Generating")
    while remaining > 0:
        bs = min(batch_size, remaining)
        x_0 = torch.randn(bs, *shape, device=device)
        samples = sampler_fn(x_0)
        if isinstance(samples, dict):
            samples = samples['x_final']
        all_samples.append(samples.cpu())
        remaining -= bs
        pbar.update(bs)
    pbar.close()
    return torch.cat(all_samples, dim=0)[:num_samples]


# ============================================================
# Train DDIM models
# ============================================================

def train_ddim_models(args, device):
    """Train DDIM cloud and edge models with the same UNet architecture."""
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    for model_type in ['cloud', 'edge']:
        print(f"\n{'='*60}")
        print(f"Training DDIM {model_type} model")
        print(f"{'='*60}")

        model_fn = build_cloud_model if model_type == 'cloud' \
            else build_edge_model
        model = model_fn().to(device)
        print(f"  Parameters: {count_params(model)/1e6:.1f}M")

        trainer = DDIMTrainer(model, num_timesteps=1000)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)

        best_loss = float('inf')
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
            for x, _ in pbar:
                x = x.to(device)
                loss = trainer.compute_loss(x)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
                if num_batches % 100 == 0:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

            scheduler.step()
            avg_loss = epoch_loss / num_batches
            print(f"  Epoch {epoch}: loss={avg_loss:.4f}, "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'framework': 'ddim',
            }
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ckpt, os.path.join(
                    args.save_dir, f'ddim_{model_type}_best.pt'))
                print(f"  -> Saved best (loss={avg_loss:.4f})")
            if epoch % 10 == 0:
                torch.save(ckpt, os.path.join(
                    args.save_dir, f'ddim_{model_type}_epoch{epoch}.pt'))

    print("\nDDIM training complete.")


# ============================================================
# Exp6: DC on DDIM vs DC on RF
# ============================================================

def run_exp6(args, device):
    """
    Core comparison: same DC protocol, RF vs DDIM.

    Methods tested for each framework:
      - Cloud Only (upper bound)
      - Edge Only (lower bound)
      - State Relay (t*=0.5)
      - DC single-point (t*=0.7)
      - DC 3-point (0.3, 0.6, 0.8)
      - DC 3-point + compression (50% sparse, 4-bit)
    """
    print("\n" + "=" * 70)
    print("EXP6: Direction Correction — RF vs DDIM")
    print("=" * 70)

    # Load RF models
    print("\n--- Loading RF models ---")
    rf_cloud = load_model(build_cloud_model, args.rf_cloud_ckpt, device)
    rf_edge = load_model(build_edge_model, args.rf_edge_ckpt, device)
    rf_collab = DirectionCorrectionSampler(rf_edge, rf_cloud)

    # Load DDIM models
    print("\n--- Loading DDIM models ---")
    ddim_cloud = load_model(build_cloud_model, args.ddim_cloud_ckpt, device)
    ddim_edge = load_model(build_edge_model, args.ddim_edge_ckpt, device)
    ddim_collab = DCOnDDIMSampler(ddim_edge, ddim_cloud, num_timesteps=1000)

    shape = (3, 32, 32)
    ref_dir = prepare_cifar10_reference()
    compress_fn = make_compress_fn(keep_ratio=0.5, bits=4)

    # Define methods for each framework
    rf_methods = {
        'Cloud Only': lambda x0: rf_collab.sample_cloud_only(
            x0, num_steps=50),
        'Edge Only': lambda x0: rf_collab.sample_edge_only(
            x0, num_steps=20),
        'State Relay (t*=0.5)': lambda x0: rf_collab.sample_state_relay(
            x0, t_star=0.5, edge_steps=10, cloud_steps=25),
        'DC 1-point (t*=0.7)': lambda x0:
            rf_collab.sample_direction_correction(
                x0, t_star=0.7, edge_steps_before=10, edge_steps_after=10),
        'DC 3-point': lambda x0:
            rf_collab.sample_multi_point_correction(
                x0, query_points=[0.3, 0.6, 0.8], total_steps=20),
        'DC 3-point + compress': lambda x0:
            rf_collab.sample_multi_point_correction(
                x0, query_points=[0.3, 0.6, 0.8], total_steps=20,
                compress_fn=compress_fn),
    }

    ddim_methods = {
        'Cloud Only': lambda x0: ddim_collab.sample_cloud_only(
            x0, num_steps=50),
        'Edge Only': lambda x0: ddim_collab.sample_edge_only(
            x0, num_steps=20),
        'State Relay (t*=0.5)': lambda x0: ddim_collab.sample_state_relay(
            x0, t_star=0.5, edge_steps=10, cloud_steps=25),
        'DC 1-point (t*=0.7)': lambda x0:
            ddim_collab.sample_direction_correction(
                x0, t_star=0.7, edge_steps_before=10, edge_steps_after=10),
        'DC 3-point': lambda x0:
            ddim_collab.sample_multi_point_correction(
                x0, query_points=[0.3, 0.6, 0.8], total_steps=20),
        'DC 3-point + compress': lambda x0:
            ddim_collab.sample_multi_point_correction(
                x0, query_points=[0.3, 0.6, 0.8], total_steps=20,
                compress_fn=compress_fn),
    }

    results = {'rf': {}, 'ddim': {}}

    for framework, methods, label in [
        ('rf', rf_methods, 'Rectified Flow'),
        ('ddim', ddim_methods, 'DDIM'),
    ]:
        print(f"\n{'='*50}")
        print(f"  Framework: {label}")
        print(f"{'='*50}")

        for method_name, sampler_fn in methods.items():
            print(f"\n  [{method_name}]")
            torch.manual_seed(42)

            samples = generate_samples(
                sampler_fn, args.num_samples, args.batch_size, shape, device)

            tag = f"{framework}_{method_name.replace(' ', '_')}"
            gen_dir = save_samples_to_dir(
                samples, f"{args.output_dir}/exp6/{tag}/")
            fid = compute_fid(gen_dir, ref_dir, device=device)

            results[framework][method_name] = fid
            print(f"    FID = {fid:.2f}")

    # Save results
    out_path = os.path.join(args.output_dir, "exp6_rf_vs_ddim.json")
    with open(out_path, 'w') as f:
        json.dump({
            'results': results,
            'num_samples': args.num_samples,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Method':<30s} {'RF FID':>10s} {'DDIM FID':>10s} {'Δ':>8s}")
    print("-" * 60)
    for method in rf_methods:
        rf_fid = results['rf'].get(method, float('nan'))
        ddim_fid = results['ddim'].get(method, float('nan'))
        delta = ddim_fid - rf_fid
        print(f"{method:<30s} {rf_fid:>10.2f} {ddim_fid:>10.2f} "
              f"{delta:>+8.2f}")
    print("=" * 70)


# ============================================================
# Exp7: Trajectory Straightness
# ============================================================

def run_exp7(args, device):
    """Compare trajectory straightness: RF vs DDIM."""
    print("\n" + "=" * 70)
    print("EXP7: Trajectory Straightness Score (TSS)")
    print("=" * 70)

    results = {}
    shape = (3, 32, 32)

    for framework, ckpt_cloud, ckpt_edge, label in [
        ('rf', args.rf_cloud_ckpt, args.rf_edge_ckpt, 'Rectified Flow'),
        ('ddim', args.ddim_cloud_ckpt, args.ddim_edge_ckpt, 'DDIM'),
    ]:
        print(f"\n--- {label} ---")
        results[framework] = {}

        for model_type, ckpt_path in [
            ('cloud', ckpt_cloud), ('edge', ckpt_edge)
        ]:
            print(f"  [{model_type}]")
            model_fn = build_cloud_model if model_type == 'cloud' \
                else build_edge_model
            model = load_model(model_fn, ckpt_path, device)

            torch.manual_seed(42)
            x_init = torch.randn(args.tss_batch_size, *shape, device=device)

            tss = compute_trajectory_straightness(
                model, x_init, sampler_type=framework,
                num_steps=100, num_timesteps=1000)

            results[framework][model_type] = tss
            print(f"    TSS = {tss['tss_mean']:.4f} ± {tss['tss_std']:.4f}")
            print(f"    Chord = {tss['chord_mean']:.2f}, "
                  f"Arc = {tss['arc_length_mean']:.2f}")

    out_path = os.path.join(args.output_dir, "exp7_straightness.json")
    with open(out_path, 'w') as f:
        json.dump({
            'results': {fw: {mt: {k: v for k, v in d.items()
                                   if k != 'tss_per_sample'}
                              for mt, d in fwd.items()}
                        for fw, fwd in results.items()},
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 50)
    print(f"{'Framework':<15s} {'Model':<8s} {'TSS':>8s}")
    print("-" * 35)
    for fw in ['rf', 'ddim']:
        for mt in ['cloud', 'edge']:
            tss_val = results[fw][mt]['tss_mean']
            print(f"{fw.upper():<15s} {mt:<8s} {tss_val:>8.4f}")
    print("=" * 50)


# ============================================================
# Exp8: δv / δε temporal consistency
# ============================================================

def run_exp8(args, device):
    """Compare δv consistency (RF) vs δε consistency (DDIM) over time."""
    print("\n" + "=" * 70)
    print("EXP8: Correction Vector Temporal Consistency")
    print("=" * 70)

    shape = (3, 32, 32)
    results = {}

    for framework, ckpt_cloud, ckpt_edge, label in [
        ('rf', args.rf_cloud_ckpt, args.rf_edge_ckpt, 'Rectified Flow'),
        ('ddim', args.ddim_cloud_ckpt, args.ddim_edge_ckpt, 'DDIM'),
    ]:
        print(f"\n--- {label} ---")

        model_fn_cloud = build_cloud_model
        model_fn_edge = build_edge_model

        cloud = load_model(model_fn_cloud, ckpt_cloud, device)
        edge = load_model(model_fn_edge, ckpt_edge, device)

        torch.manual_seed(42)
        x_init = torch.randn(args.tss_batch_size, *shape, device=device)

        consistency = compute_delta_consistency(
            edge, cloud, x_init,
            t_star=0.5, sampler_type=framework,
            num_steps=50, num_probe_points=9, num_timesteps=1000)

        results[framework] = consistency

        print(f"  t* = {consistency['t_star']}")
        for t, vals in sorted(consistency['consistency'].items()):
            bar = '█' * int(vals['relative_diff_mean'] * 20)
            print(f"  t={t:.1f}: relative_diff = "
                  f"{vals['relative_diff_mean']:.4f} {bar}")

    out_path = os.path.join(args.output_dir, "exp8_consistency.json")
    with open(out_path, 'w') as f:
        json.dump({
            'results': results,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


# ============================================================
# Exp9: δv / δε spectrum analysis
# ============================================================

def run_exp9(args, device):
    """SVD analysis of δv (RF) vs δε (DDIM)."""
    print("\n" + "=" * 70)
    print("EXP9: Correction Vector Spectrum Analysis")
    print("=" * 70)

    shape = (3, 32, 32)
    results = {}

    for framework, ckpt_cloud, ckpt_edge, label in [
        ('rf', args.rf_cloud_ckpt, args.rf_edge_ckpt, 'Rectified Flow'),
        ('ddim', args.ddim_cloud_ckpt, args.ddim_edge_ckpt, 'DDIM'),
    ]:
        print(f"\n--- {label} ---")

        cloud = load_model(build_cloud_model, ckpt_cloud, device)
        edge = load_model(build_edge_model, ckpt_edge, device)

        torch.manual_seed(42)
        # Use more samples for meaningful SVD
        x_init = torch.randn(min(args.tss_batch_size, 256), *shape,
                              device=device)

        spectrum = compute_delta_spectrum(
            edge, cloud, x_init,
            t_star=0.5, sampler_type=framework,
            num_steps=50, top_k=50, num_timesteps=1000)

        results[framework] = spectrum

        print(f"  Feature dim: {spectrum['feature_dim']}")
        print(f"  k for 90% energy: {spectrum['k_for_90pct']}")
        print(f"  k for 95% energy: {spectrum['k_for_95pct']}")
        print(f"  k for 99% energy: {spectrum['k_for_99pct']}")
        print(f"  Top-5 singular values: "
              f"{[f'{s:.2f}' for s in spectrum['singular_values'][:5]]}")

    out_path = os.path.join(args.output_dir, "exp9_spectrum.json")
    with open(out_path, 'w') as f:
        json.dump({
            'results': results,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 50)
    print(f"{'Metric':<25s} {'RF':>10s} {'DDIM':>10s}")
    print("-" * 47)
    for metric in ['k_for_90pct', 'k_for_95pct', 'k_for_99pct']:
        rf_val = results['rf'][metric]
        ddim_val = results['ddim'][metric]
        label = metric.replace('k_for_', 'k @ ').replace('pct', '%')
        print(f"{label:<25s} {rf_val:>10d} {ddim_val:>10d}")
    print("=" * 50)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Why Flow Models? — Controlled comparison experiments")

    parser.add_argument('--mode', type=str, required=True,
                        choices=['train_ddim', 'exp6', 'exp7', 'exp8',
                                 'exp9', 'all'],
                        help='Which experiment to run')

    # Model checkpoints
    parser.add_argument('--rf_cloud_ckpt', type=str,
                        default='checkpoints/cloud_best.pt')
    parser.add_argument('--rf_edge_ckpt', type=str,
                        default='checkpoints/edge_best.pt')
    parser.add_argument('--ddim_cloud_ckpt', type=str,
                        default='checkpoints/ddim_cloud_best.pt')
    parser.add_argument('--ddim_edge_ckpt', type=str,
                        default='checkpoints/ddim_edge_best.pt')

    # Training params (for train_ddim)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--save_dir', type=str, default='checkpoints/')

    # Experiment params
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Samples for FID computation')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--tss_batch_size', type=int, default=64,
                        help='Batch size for trajectory/consistency analysis')
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == 'train_ddim':
        train_ddim_models(args, device)
    elif args.mode == 'exp6':
        run_exp6(args, device)
    elif args.mode == 'exp7':
        run_exp7(args, device)
    elif args.mode == 'exp8':
        run_exp8(args, device)
    elif args.mode == 'exp9':
        run_exp9(args, device)
    elif args.mode == 'all':
        run_exp7(args, device)
        run_exp8(args, device)
        run_exp9(args, device)
        run_exp6(args, device)


if __name__ == "__main__":
    main()
