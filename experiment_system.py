"""
experiment_system.py — System-level experiments for INFOCOM submission.

Exp6: End-to-end latency breakdown
Exp7: AdaDC vs static strategies under varying bandwidth
Exp8: Pareto frontier comparison (FID vs comm, FID vs latency)
Exp9: Scalability analysis (varying model sizes)
"""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm

from models import build_cloud_model, build_edge_model, count_params
from rectified_flow import DirectionCorrectionSampler
from compression import make_compress_fn, compute_transmitted_size_kb
from metrics import compute_fid, save_samples_to_dir, prepare_cifar10_reference
from network_model import (NetworkProfile, NetworkSimulator,
                           BandwidthSwitchSimulator, NETWORK_PARAMS)
from latency_profiler import (profile_system, SystemLatencyProfile,
                               SIMULATED_PROFILES)
from adadc_protocol import (ParetoFrontier, AdaDCProtocol, AdaDCConfig,
                             build_protocol)
from baselines import (early_exit_samples, split_inference_latency,
                        multi_relay_samples, communication_cost_table)


def load_model(model_fn, ckpt_path, device):
    model = model_fn()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    print(f"  Loaded {ckpt_path} (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})")
    return model


def generate_samples(sampler_fn, num_samples, batch_size, shape, device):
    """Generate samples in batches."""
    all_samples = []
    generated = 0
    pbar = tqdm(total=num_samples, desc="Generating")
    while generated < num_samples:
        bs = min(batch_size, num_samples - generated)
        x_0 = torch.randn(bs, *shape, device=device)
        result = sampler_fn(x_0)
        if isinstance(result, dict):
            samples = result['x_final']
        else:
            samples = result
        all_samples.append(samples.cpu())
        generated += bs
        pbar.update(bs)
    pbar.close()
    return torch.cat(all_samples, dim=0)[:num_samples]


# ============================================================
# Experiment 6: End-to-End Latency Breakdown
# ============================================================

def run_exp6_latency_breakdown(args, device):
    """
    For each method × each network condition, compute latency breakdown:
    edge_compute + cloud_compute + uplink + downlink.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: End-to-End Latency Breakdown")
    print("=" * 60)

    cloud = load_model(build_cloud_model, args.cloud_ckpt, device)
    edge = load_model(build_edge_model, args.edge_ckpt, device)

    # Profile actual computation latency
    print("\nProfiling model latency...")
    latency = profile_system(edge, cloud, input_shape=(1, 3, 32, 32), device=device)
    print(f"  Edge step: {latency.edge_step_ms:.2f} ms")
    print(f"  Cloud step: {latency.cloud_step_ms:.2f} ms")

    # Build Pareto frontier
    pareto = ParetoFrontier().build_from_results(args.results_dir)

    results = {}
    methods = ['edge_only', 'cloud_only', 'state_relay']

    # Add DC configs from Pareto frontier
    dc_configs = [
        ('DC (1pt, kr=10%, 4bit)', AdaDCConfig(
            num_queries=1, query_points=[0.7],
            keep_ratio=0.1, bits=4, comm_kb=0.76)),
        ('DC (1pt, kr=50%, 4bit)', AdaDCConfig(
            num_queries=1, query_points=[0.7],
            keep_ratio=0.5, bits=4, comm_kb=3.76)),
        ('DC (3pt, kr=20%, 4bit)', AdaDCConfig(
            num_queries=3, query_points=[0.3, 0.6, 0.8],
            keep_ratio=0.2, bits=4, comm_kb=4.52)),
    ]

    for profile in NetworkProfile:
        print(f"\n--- Network: {profile.value} ---")
        protocol = AdaDCProtocol(pareto, latency,
                                  NetworkSimulator(profile, seed=42))

        profile_results = {}

        for method in methods:
            lat = protocol.compute_method_latency(method, total_steps=20)
            profile_results[method] = lat
            print(f"  {lat.get('method', method):30s}: "
                  f"total={lat['total_ms']:.1f}ms "
                  f"(edge={lat['edge_compute_ms']:.1f}, "
                  f"cloud={lat['cloud_compute_ms']:.1f}, "
                  f"up={lat['uplink_ms']:.1f}, "
                  f"down={lat['downlink_ms']:.1f})")

        for label, config in dc_configs:
            lat = protocol.estimate_total_latency(config, total_steps=20)
            lat['method'] = label
            lat['comm_kb'] = config.comm_kb
            profile_results[label] = lat
            print(f"  {label:30s}: "
                  f"total={lat['total_ms']:.1f}ms "
                  f"(edge={lat['edge_compute_ms']:.1f}, "
                  f"cloud={lat['cloud_compute_ms']:.1f}, "
                  f"up={lat['uplink_ms']:.1f}, "
                  f"down={lat['downlink_ms']:.1f})")

        results[profile.value] = profile_results

    # Save
    out_path = os.path.join(args.output_dir, 'exp6_latency_breakdown.json')
    # Convert for JSON serialization
    serializable = {}
    for net, methods_dict in results.items():
        serializable[net] = {}
        for method, lat in methods_dict.items():
            serializable[net][method] = {k: float(v) if isinstance(v, (int, float, np.floating))
                                          else v
                                          for k, v in lat.items()}
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


# ============================================================
# Experiment 7: AdaDC vs Static Strategies
# ============================================================

def run_exp7_adadc_vs_static(args, device):
    """
    Compare AdaDC (adaptive) vs static DC strategies under varying bandwidth.

    Scenarios:
    - Constant bandwidth (4G)
    - Bandwidth drop (4G → 3G at t=0.5)
    - Bandwidth spike (3G → WiFi at t=0.5)
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 7: AdaDC vs Static Strategies")
    print("=" * 60)

    cloud = load_model(build_cloud_model, args.cloud_ckpt, device)
    edge = load_model(build_edge_model, args.edge_ckpt, device)
    collab = DirectionCorrectionSampler(edge, cloud)

    latency = profile_system(edge, cloud, input_shape=(1, 3, 32, 32), device=device)
    pareto = ParetoFrontier().build_from_results(args.results_dir)

    # Prepare reference for FID
    ref_dir = prepare_cifar10_reference()

    scenarios = [
        ('Constant 4G', NetworkSimulator(NetworkProfile.G4, seed=42)),
        ('4G→3G drop', BandwidthSwitchSimulator(
            NetworkProfile.G4, NetworkProfile.G3, t_switch=0.5, seed=42)),
        ('3G→WiFi spike', BandwidthSwitchSimulator(
            NetworkProfile.G3, NetworkProfile.WIFI, t_switch=0.5, seed=42)),
        ('Constant 3G', NetworkSimulator(NetworkProfile.G3, seed=42)),
    ]

    # Static strategies
    static_configs = {
        'Edge Only': AdaDCConfig(num_queries=0, query_points=[],
                                  comm_kb=0, label='Edge Only'),
        'Static 1pt (kr=10%)': AdaDCConfig(
            num_queries=1, query_points=[0.7],
            keep_ratio=0.1, bits=4, comm_kb=0.76,
            label='Static 1pt'),
        'Static 3pt (kr=20%)': AdaDCConfig(
            num_queries=3, query_points=[0.3, 0.6, 0.8],
            keep_ratio=0.2, bits=4, comm_kb=4.52,
            label='Static 3pt'),
    }

    latency_budgets = [500, 1000, 2000, 5000]  # ms

    results = {}

    for scenario_name, network_sim in scenarios:
        print(f"\n--- Scenario: {scenario_name} ---")
        scenario_results = {}

        # Static strategies: generate samples and compute FID
        for name, config in static_configs.items():
            if config.num_queries == 0:
                sampler_fn = lambda x0: collab.sample_edge_only(x0, num_steps=20)
            elif config.num_queries == 1:
                cfn = make_compress_fn(config.keep_ratio, config.bits)
                sampler_fn = lambda x0, cf=cfn: collab.sample_direction_correction(
                    x0, t_star=0.7, edge_steps_before=14,
                    edge_steps_after=6, compress_fn=cf)
            else:
                cfn = make_compress_fn(config.keep_ratio, config.bits)
                sampler_fn = lambda x0, cf=cfn: collab.sample_multi_point_correction(
                    x0, query_points=config.query_points,
                    total_steps=20, compress_fn=cf)

            samples = generate_samples(
                sampler_fn, args.num_samples, args.batch_size,
                (3, 32, 32), device)
            save_dir = os.path.join(args.output_dir, 'exp7',
                                     scenario_name.replace(' ', '_'),
                                     name.replace(' ', '_'))
            save_samples_to_dir(samples, save_dir)
            fid = compute_fid(save_dir, ref_dir, device=str(device))

            # Compute latency under this scenario
            protocol = AdaDCProtocol(pareto, latency, network_sim)
            if config.num_queries > 0:
                lat = protocol.estimate_total_latency(config, total_steps=20)
                total_ms = lat['total_ms']
            else:
                total_ms = 20 * latency.edge_step_ms

            scenario_results[name] = {
                'fid': fid,
                'comm_kb': config.comm_kb,
                'latency_ms': total_ms,
            }
            print(f"  {name:25s}: FID={fid:.2f}, comm={config.comm_kb:.2f}KB, "
                  f"latency={total_ms:.1f}ms")

        # AdaDC: adaptive strategy
        for budget in latency_budgets:
            protocol = AdaDCProtocol(pareto, latency, network_sim)
            config = protocol.select_strategy_offline(
                latency_budget_ms=budget, total_steps=20)

            if config.num_queries == 0:
                sampler_fn = lambda x0: collab.sample_edge_only(x0, num_steps=20)
            elif config.num_queries == 1:
                cfn = make_compress_fn(config.keep_ratio, config.bits)
                sampler_fn = lambda x0, cf=cfn: collab.sample_direction_correction(
                    x0, t_star=config.query_points[0],
                    edge_steps_before=int(20 * config.query_points[0]),
                    edge_steps_after=20 - int(20 * config.query_points[0]),
                    compress_fn=cf)
            else:
                cfn = make_compress_fn(config.keep_ratio, config.bits)
                sampler_fn = lambda x0, cf=cfn, qp=config.query_points: \
                    collab.sample_multi_point_correction(
                        x0, query_points=qp, total_steps=20, compress_fn=cf)

            samples = generate_samples(
                sampler_fn, args.num_samples, args.batch_size,
                (3, 32, 32), device)
            save_dir = os.path.join(args.output_dir, 'exp7',
                                     scenario_name.replace(' ', '_'),
                                     f'adadc_budget{budget}')
            save_samples_to_dir(samples, save_dir)
            fid = compute_fid(save_dir, ref_dir, device=str(device))

            lat = protocol.estimate_total_latency(config, total_steps=20)
            ada_name = f'AdaDC (budget={budget}ms)'
            scenario_results[ada_name] = {
                'fid': fid,
                'comm_kb': config.comm_kb,
                'latency_ms': lat['total_ms'],
                'selected_config': str(config),
            }
            print(f"  {ada_name:25s}: FID={fid:.2f}, comm={config.comm_kb:.2f}KB, "
                  f"latency={lat['total_ms']:.1f}ms, config={config.label}")

        results[scenario_name] = scenario_results

    out_path = os.path.join(args.output_dir, 'exp7_adadc_vs_static.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    return results


# ============================================================
# Experiment 8: Pareto Frontier Comparison
# ============================================================

def run_exp8_pareto(args, device):
    """
    Build and visualize the complete Pareto frontier:
    FID vs Communication (KB) and FID vs Latency (ms) under each network.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 8: Pareto Frontier Comparison")
    print("=" * 60)

    cloud = load_model(build_cloud_model, args.cloud_ckpt, device)
    edge = load_model(build_edge_model, args.edge_ckpt, device)

    latency = profile_system(edge, cloud, input_shape=(1, 3, 32, 32), device=device)
    pareto = ParetoFrontier().build_from_results(args.results_dir)

    print(f"\nLoaded {len(pareto.all_configs)} data points, "
          f"{len(pareto.pareto_configs)} Pareto-optimal")

    # FID vs Communication table
    print("\n--- Pareto Frontier (FID vs Communication) ---")
    print(f"{'Config':40s} {'FID':>8s} {'Comm(KB)':>10s}")
    print("-" * 60)
    for cfg in pareto.pareto_configs:
        print(f"{cfg.label:40s} {cfg.fid:8.2f} {cfg.comm_kb:10.2f}")

    # FID vs Latency for each network
    results = {
        'pareto_points': [(c.fid, c.comm_kb, c.label)
                           for c in pareto.pareto_configs],
        'all_points': [(c.fid, c.comm_kb, c.label, c.num_queries)
                        for c in pareto.all_configs],
        'latency_by_network': {},
    }

    for profile in NetworkProfile:
        print(f"\n--- FID vs Latency under {profile.value} ---")
        protocol = AdaDCProtocol(pareto, latency,
                                  NetworkSimulator(profile, seed=42))
        network_results = []
        for cfg in pareto.all_configs:
            if cfg.num_queries > 0:
                lat = protocol.estimate_total_latency(cfg, total_steps=20)
                total_ms = lat['total_ms']
            else:
                total_ms = 20 * latency.edge_step_ms
            network_results.append({
                'label': cfg.label,
                'fid': cfg.fid,
                'comm_kb': cfg.comm_kb,
                'latency_ms': total_ms,
                'num_queries': cfg.num_queries,
            })
        results['latency_by_network'][profile.value] = network_results

        # Print top-5 by FID
        sorted_by_fid = sorted(network_results, key=lambda x: x['fid'])
        for r in sorted_by_fid[:5]:
            print(f"  {r['label']:35s}: FID={r['fid']:.1f}, "
                  f"latency={r['latency_ms']:.1f}ms, comm={r['comm_kb']:.2f}KB")

    # Add state relay and cloud-only reference points
    try:
        with open(os.path.join(args.results_dir, 'experiment1_results.json')) as f:
            exp1 = json.load(f)
        results['reference_points'] = {
            'state_relay': {
                'fid': exp1['fid_scores'].get('State Relay', 36.8),
                'comm_kb': 24.0,  # up + down
            },
            'cloud_only': {
                'fid': exp1['fid_scores'].get('Cloud Only (ref)', 27.8),
                'comm_kb': 24.0,
            },
        }
    except FileNotFoundError:
        pass

    out_path = os.path.join(args.output_dir, 'exp8_pareto.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    return results


# ============================================================
# Experiment 9: Scalability Analysis
# ============================================================

def run_exp9_scalability(args, device):
    """
    Analyze how DC advantage scales with model size.
    Uses simulated latency profiles without retraining.

    Key insight: as models get larger, compute time increases but
    delta_v communication stays small, making DC increasingly attractive.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 9: Scalability Analysis")
    print("=" * 60)

    # Real measurement as baseline
    cloud = load_model(build_cloud_model, args.cloud_ckpt, device)
    edge = load_model(build_edge_model, args.edge_ckpt, device)
    real_latency = profile_system(edge, cloud, input_shape=(1, 3, 32, 32), device=device)

    # Scale factors
    scale_configs = [
        ('Current (4M/35M)', 1, 1),
        ('Medium (20M/175M)', 5, 5),
        ('Large (80M/700M)', 20, 20),
        ('XL (200M/1.75B)', 50, 50),
        ('SD-scale (500M/3.5B)', 125, 100),
    ]

    # Network conditions
    networks = {
        '3G': NetworkSimulator(NetworkProfile.G3, seed=42),
        '4G': NetworkSimulator(NetworkProfile.G4, seed=42),
        'WiFi': NetworkSimulator(NetworkProfile.WIFI, seed=42),
        '5G': NetworkSimulator(NetworkProfile.G5, seed=42),
    }

    # Communication sizes (fixed regardless of model size for same resolution)
    state_kb = 12.0  # CIFAR-10 full state
    dc_compressed_kb = 0.76  # DC with kr=10%, 4-bit

    pareto = ParetoFrontier().build_from_results(args.results_dir)

    results = {}

    for net_name, network_sim in networks.items():
        print(f"\n--- Network: {net_name} ---")
        net_results = []

        for scale_name, edge_scale, cloud_scale in scale_configs:
            scaled = real_latency.scale_to_params(
                real_latency.edge_params * edge_scale,
                real_latency.cloud_params * cloud_scale)

            # State Relay latency
            protocol_sr = AdaDCProtocol(pareto, scaled, network_sim)
            sr_lat = protocol_sr.compute_method_latency(
                'state_relay', total_steps=20)

            # DC latency (1-point, compressed)
            dc_config = AdaDCConfig(
                num_queries=1, query_points=[0.7],
                keep_ratio=0.1, bits=4, comm_kb=dc_compressed_kb)
            dc_lat = protocol_sr.estimate_total_latency(dc_config, total_steps=20)

            # Edge-only latency
            eo_lat = protocol_sr.compute_method_latency('edge_only', total_steps=20)

            # DC advantage: latency reduction vs state relay
            speedup = sr_lat['total_ms'] / dc_lat['total_ms'] if dc_lat['total_ms'] > 0 else 1

            entry = {
                'scale': scale_name,
                'edge_params_M': scaled.edge_params / 1e6,
                'cloud_params_M': scaled.cloud_params / 1e6,
                'edge_only_ms': eo_lat['total_ms'],
                'state_relay_ms': sr_lat['total_ms'],
                'dc_compressed_ms': dc_lat['total_ms'],
                'dc_speedup_vs_relay': speedup,
                'dc_comm_kb': dc_compressed_kb,
                'relay_comm_kb': 24.0,
            }
            net_results.append(entry)
            print(f"  {scale_name:25s}: EdgeOnly={eo_lat['total_ms']:.0f}ms, "
                  f"Relay={sr_lat['total_ms']:.0f}ms, "
                  f"DC={dc_lat['total_ms']:.0f}ms, "
                  f"speedup={speedup:.2f}x")

        results[net_name] = net_results

    out_path = os.path.join(args.output_dir, 'exp9_scalability.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Also print communication cost comparison table
    print("\n--- Communication Cost Table ---")
    comm_table = communication_cost_table()
    for method, info in comm_table.items():
        print(f"  {method:30s}: {info['description']}")

    return results


# ============================================================
# Run All System Experiments
# ============================================================

def run_all_system_experiments(args, device):
    """Run all system experiments sequentially."""
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    results['exp6'] = run_exp6_latency_breakdown(args, device)
    results['exp8'] = run_exp8_pareto(args, device)
    results['exp9'] = run_exp9_scalability(args, device)

    # Exp7 is expensive (generates samples), run separately if needed
    if args.run_exp7:
        results['exp7'] = run_exp7_adadc_vs_static(args, device)

    print("\n" + "=" * 60)
    print("ALL SYSTEM EXPERIMENTS COMPLETE")
    print("=" * 60)
    return results


def main():
    parser = argparse.ArgumentParser(description='System-level experiments')
    parser.add_argument('--cloud_ckpt', type=str,
                        default='checkpoints/cloud_best.pt')
    parser.add_argument('--edge_ckpt', type=str,
                        default='checkpoints/edge_best.pt')
    parser.add_argument('--results_dir', type=str, default='results/',
                        help='Directory with existing experiment results')
    parser.add_argument('--output_dir', type=str, default='results/system/',
                        help='Output directory for system experiments')
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')

    # Experiment selection
    parser.add_argument('--exp6', action='store_true', help='Run Exp6 only')
    parser.add_argument('--exp7', action='store_true', help='Run Exp7 only')
    parser.add_argument('--exp8', action='store_true', help='Run Exp8 only')
    parser.add_argument('--exp9', action='store_true', help='Run Exp9 only')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--run_exp7', action='store_true',
                        help='Include Exp7 in --all (expensive)')

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    if args.exp6:
        run_exp6_latency_breakdown(args, device)
    elif args.exp7:
        run_exp7_adadc_vs_static(args, device)
    elif args.exp8:
        run_exp8_pareto(args, device)
    elif args.exp9:
        run_exp9_scalability(args, device)
    elif args.all:
        run_all_system_experiments(args, device)
    else:
        print("Usage: python experiment_system.py --exp6|--exp7|--exp8|--exp9|--all")
        print("  --exp6: Latency breakdown (fast, no FID)")
        print("  --exp7: AdaDC vs static (slow, generates samples)")
        print("  --exp8: Pareto frontier (fast, uses existing results)")
        print("  --exp9: Scalability analysis (fast, simulated)")
        print("  --all:  Run exp6+exp8+exp9 (add --run_exp7 for exp7)")


if __name__ == '__main__':
    main()
