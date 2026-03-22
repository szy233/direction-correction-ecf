"""
experiment.py — Core experiments: Direction Correction vs State Relay vs Edge-Only.

Experiment 1: Fixed t*, compare FID of three methods
Experiment 3: Compression rate sweep (when --compression_sweep is set)

Usage:
    python experiment.py --cloud_ckpt checkpoints/cloud_best.pt \
                         --edge_ckpt checkpoints/edge_best.pt \
                         --t_star 0.5 --num_samples 10000

    python experiment.py --cloud_ckpt checkpoints/cloud_best.pt \
                         --edge_ckpt checkpoints/edge_best.pt \
                         --t_star 0.5 --compression_sweep
"""

import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

from models import build_cloud_model, build_edge_model
from rectified_flow import DirectionCorrectionSampler
from compression import make_compress_fn, compute_compression_ratio
from metrics import save_samples_to_dir, compute_fid, prepare_cifar10_reference


def load_model(model_fn, ckpt_path, device):
    """Load a trained model from checkpoint."""
    model = model_fn()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    print(f"  Loaded {ckpt_path} (epoch {ckpt.get('epoch', '?')}, loss {ckpt.get('loss', '?'):.4f})")
    return model


def generate_samples(sampler_fn, num_samples, batch_size, shape, device):
    """Generate samples in batches."""
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


def run_experiment_1(args, device):
    """
    Experiment 1: Direction Correction vs State Relay vs Edge-Only.

    For a fixed t*, compare:
      (a) State relay: edge → t* → cloud finishes (baseline, full x_{t*} transmitted)
      (b) Direction correction: edge runs full path, cloud sends δv (our method)
      (c) Edge only: edge runs everything alone (no communication)
      (d) Cloud only: cloud runs everything (quality upper bound)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Direction Correction vs State Relay vs Edge-Only")
    print(f"  t* = {args.t_star}, num_samples = {args.num_samples}")
    print("=" * 70)

    # Load models
    cloud_model = load_model(build_cloud_model, args.cloud_ckpt, device)
    edge_model = load_model(build_edge_model, args.edge_ckpt, device)

    collab = DirectionCorrectionSampler(edge_model, cloud_model)
    shape = (3, 32, 32)  # CIFAR-10
    results = {}

    # Shared initial noise for fair comparison
    torch.manual_seed(42)

    # --- (a) State Relay ---
    print("\n[a] State Relay (edge → t* → cloud finishes)")
    samples_relay = generate_samples(
        lambda x0: collab.sample_state_relay(x0, t_star=args.t_star,
                                              edge_steps=args.edge_steps,
                                              cloud_steps=args.cloud_steps),
        args.num_samples, args.batch_size, shape, device
    )
    relay_dir = save_samples_to_dir(samples_relay, f"{args.output_dir}/relay/")
    print(f"  Saved to {relay_dir}")

    # --- (b) Direction Correction (ours) ---
    print(f"\n[b] Direction Correction (ours, t*={args.t_star})")
    torch.manual_seed(42)
    alpha_values_all = []

    def dc_sampler(x0):
        result = collab.sample_direction_correction(
            x0, t_star=args.t_star,
            edge_steps_before=args.edge_steps,
            edge_steps_after=args.edge_steps,
        )
        alpha_values_all.extend(result['alpha_values'])
        return result

    samples_dc = generate_samples(dc_sampler, args.num_samples, args.batch_size, shape, device)
    dc_dir = save_samples_to_dir(samples_dc, f"{args.output_dir}/direction_correction/")
    print(f"  Saved to {dc_dir}")
    print(f"  Mean α(t) over trajectory: {np.mean(alpha_values_all):.3f}")

    # --- (c) Edge Only ---
    print("\n[c] Edge Only (no communication)")
    torch.manual_seed(42)
    samples_edge = generate_samples(
        lambda x0: collab.sample_edge_only(x0, num_steps=args.edge_steps * 2),
        args.num_samples, args.batch_size, shape, device
    )
    edge_dir = save_samples_to_dir(samples_edge, f"{args.output_dir}/edge_only/")
    print(f"  Saved to {edge_dir}")

    # --- (d) Cloud Only (reference) ---
    print("\n[d] Cloud Only (quality upper bound)")
    torch.manual_seed(42)
    samples_cloud = generate_samples(
        lambda x0: collab.sample_cloud_only(x0, num_steps=args.cloud_steps * 2),
        args.num_samples, args.batch_size, shape, device
    )
    cloud_dir = save_samples_to_dir(samples_cloud, f"{args.output_dir}/cloud_only/")
    print(f"  Saved to {cloud_dir}")

    # --- Compute FID ---
    print("\nPreparing CIFAR-10 reference images...")
    ref_dir = prepare_cifar10_reference()

    print("\nComputing FID scores...")
    for name, gen_dir in [
        ("Cloud Only (ref)", cloud_dir),
        ("State Relay", relay_dir),
        ("Direction Correction (ours)", dc_dir),
        ("Edge Only", edge_dir),
    ]:
        fid = compute_fid(gen_dir, ref_dir, device=device)
        results[name] = fid
        print(f"  {name:35s}: FID = {fid:.2f}")

    # --- Communication cost analysis ---
    print("\n--- Communication Cost Analysis ---")
    dummy_dv = torch.randn(1, 3, 32, 32)
    for method, desc, cost in [
        ("State Relay", "Full x_{t*}", 3 * 32 * 32 * 32 / 8 / 1024),
        ("Direction Correction (uncompressed)", "Full δv", 3 * 32 * 32 * 32 / 8 / 1024),
    ]:
        print(f"  {method}: {cost:.2f} KB")

    for kr in [0.05, 0.10, 0.20]:
        stats = compute_compression_ratio(dummy_dv, keep_ratio=kr, bits=8)
        print(f"  Direction Correction (sparse {kr:.0%}, 8-bit): "
              f"{stats['compressed_size_kb']:.2f} KB "
              f"({stats['compression_ratio']:.1%} of state relay)")

    # Save results
    results_path = os.path.join(args.output_dir, "experiment1_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            't_star': args.t_star,
            'num_samples': args.num_samples,
            'fid_scores': results,
            'mean_alpha': float(np.mean(alpha_values_all)),
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def run_compression_sweep(args, device):
    """
    Experiment 3: How much can we compress δv before quality degrades?

    Sweep over sparsity ratios and quantization bits.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Compression Rate Sweep")
    print("=" * 70)

    cloud_model = load_model(build_cloud_model, args.cloud_ckpt, device)
    edge_model = load_model(build_edge_model, args.edge_ckpt, device)
    collab = DirectionCorrectionSampler(edge_model, cloud_model)
    shape = (3, 32, 32)

    ref_dir = prepare_cifar10_reference()
    results = []

    keep_ratios = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
    quant_bits_list = [4, 8]

    for bits in quant_bits_list:
        for kr in keep_ratios:
            print(f"\n--- keep_ratio={kr:.0%}, bits={bits} ---")
            torch.manual_seed(42)

            compress_fn = make_compress_fn(keep_ratio=kr, bits=bits)

            samples = generate_samples(
                lambda x0: collab.sample_direction_correction(
                    x0, t_star=args.t_star,
                    edge_steps_before=args.edge_steps,
                    edge_steps_after=args.edge_steps,
                    compress_fn=compress_fn,
                ),
                args.num_samples, args.batch_size, shape, device
            )
            gen_dir = save_samples_to_dir(
                samples, f"{args.output_dir}/compress_kr{kr}_b{bits}/"
            )
            fid = compute_fid(gen_dir, ref_dir, device=device)
            stats = compute_compression_ratio(
                torch.randn(1, 3, 32, 32), keep_ratio=kr, bits=bits
            )

            entry = {
                'keep_ratio': kr,
                'bits': bits,
                'fid': fid,
                'compression_ratio': stats['compression_ratio'],
                'compressed_kb': stats['compressed_size_kb'],
            }
            results.append(entry)
            print(f"  FID = {fid:.2f}, compression = {stats['compression_ratio']:.3f}")

    # Save
    results_path = os.path.join(args.output_dir, "experiment3_compression_sweep.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'keep%':>6s} {'bits':>4s} {'ratio':>8s} {'FID':>8s}")
    print("-" * 30)
    for r in results:
        print(f"{r['keep_ratio']:6.0%} {r['bits']:4d} {r['compression_ratio']:8.3f} {r['fid']:8.2f}")


def run_multi_point(args, device):
    """
    Experiment 4: Multi-point direction correction.

    Query cloud at multiple time points instead of a single t*.
    Compare 1-point, 2-point, 3-point corrections.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Multi-Point Direction Correction")
    print("=" * 70)

    cloud_model = load_model(build_cloud_model, args.cloud_ckpt, device)
    edge_model = load_model(build_edge_model, args.edge_ckpt, device)
    collab = DirectionCorrectionSampler(edge_model, cloud_model)
    shape = (3, 32, 32)

    ref_dir = prepare_cifar10_reference()

    # Configurations: (label, query_points)
    configs = [
        ("1-point (t=0.7)",         [0.7]),
        ("2-point (0.4, 0.8)",      [0.4, 0.8]),
        ("2-point (0.3, 0.7)",      [0.3, 0.7]),
        ("2-point (0.5, 0.8)",      [0.5, 0.8]),
        ("3-point (0.3, 0.6, 0.8)", [0.3, 0.6, 0.8]),
        ("3-point (0.2, 0.5, 0.8)", [0.2, 0.5, 0.8]),
    ]

    results = []

    for label, qpts in configs:
        print(f"\n--- {label} ---")
        torch.manual_seed(42)

        samples = generate_samples(
            lambda x0, qp=qpts: collab.sample_multi_point_correction(
                x0, query_points=qp, total_steps=args.edge_steps * 2,
            ),
            args.num_samples, args.batch_size, shape, device
        )

        tag = label.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
        gen_dir = save_samples_to_dir(samples, f"{args.output_dir}/multi_point/{tag}/")
        fid = compute_fid(gen_dir, ref_dir, device=device)

        # Communication cost: each query adds one δv (12KB uncompressed for CIFAR-10)
        comm_kb = len(qpts) * 3 * 32 * 32 * 32 / 8 / 1024

        entry = {
            'label': label,
            'query_points': qpts,
            'num_queries': len(qpts),
            'fid': fid,
            'comm_kb': comm_kb,
        }
        results.append(entry)
        print(f"  FID = {fid:.2f}, queries = {len(qpts)}, comm = {comm_kb:.1f} KB")

    # Add baselines for reference
    print("\n--- Baselines ---")
    for name, sampler_fn, seed in [
        ("Edge Only", lambda x0: collab.sample_edge_only(x0, num_steps=args.edge_steps * 2), 42),
        ("State Relay (t*=0.5)", lambda x0: collab.sample_state_relay(x0, t_star=0.5, edge_steps=args.edge_steps, cloud_steps=args.cloud_steps), 42),
    ]:
        torch.manual_seed(seed)
        samples = generate_samples(sampler_fn, args.num_samples, args.batch_size, shape, device)
        tag = name.replace(" ", "_").replace("*", "").replace("(", "").replace(")", "").replace("=", "")
        gen_dir = save_samples_to_dir(samples, f"{args.output_dir}/multi_point/{tag}/")
        fid = compute_fid(gen_dir, ref_dir, device=device)
        results.append({'label': name, 'query_points': [], 'num_queries': 0, 'fid': fid, 'comm_kb': 0 if 'Edge' in name else 12.0})
        print(f"  {name}: FID = {fid:.2f}")

    # Save results
    results_path = os.path.join(args.output_dir, "experiment4_multi_point.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Method':>35s} {'Queries':>7s} {'FID':>8s} {'Comm':>8s}")
    print("-" * 60)
    for r in results:
        print(f"{r['label']:>35s} {r['num_queries']:>7d} {r['fid']:>8.2f} {r['comm_kb']:>7.1f}KB")


def main():
    parser = argparse.ArgumentParser(description="Run edge-cloud collaboration experiments")
    parser.add_argument('--cloud_ckpt', type=str, required=True)
    parser.add_argument('--edge_ckpt', type=str, required=True)
    parser.add_argument('--t_star', type=float, default=0.5)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--edge_steps', type=int, default=10,
                        help="Euler steps for edge (large step size)")
    parser.add_argument('--cloud_steps', type=int, default=50,
                        help="Euler steps for cloud (small step size)")
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--compression_sweep', action='store_true',
                        help="Run compression rate experiment instead of main experiment")
    parser.add_argument('--multi_point', action='store_true',
                        help="Run multi-point direction correction experiment")
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)

    if args.multi_point:
        run_multi_point(args, device)
    elif args.compression_sweep:
        run_compression_sweep(args, device)
    else:
        run_experiment_1(args, device)


if __name__ == "__main__":
    main()
