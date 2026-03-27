"""
experiment_v2.py — System experiments for INFOCOM submission (v2).

Experiments:
  Exp10: Protocol comparison on real traces (Hybrid vs Offline vs Greedy vs Optimal)
  Exp11: Scheduling strategy analysis (drift-aware vs periodic vs static)
  Exp12: Scalability across resolutions (CIFAR → SD-1.5 → SDXL)
  Exp13: Seed-sync ablation (naive upload vs seed-sync protocol)
  Exp14: Dynamic bandwidth adaptation (bandwidth spike/drop stress test)

All experiments produce JSON results + console summary tables.
"""

import os
import json
import argparse
import time
import numpy as np
import torch
from tqdm import tqdm

from models import build_cloud_model, build_edge_model, count_params
from rectified_flow import DirectionCorrectionSampler
from compression import make_compress_fn, compute_transmitted_size_kb
from metrics import compute_fid, save_samples_to_dir, prepare_cifar10_reference
from latency_profiler import (profile_system, SystemLatencyProfile,
                               SIMULATED_PROFILES)
from network_model import NetworkSimulator, NetworkProfile, NETWORK_PARAMS
from real_traces import (generate_3gpp_trace, generate_markov_trace,
                          get_trace_library, TraceReplaySimulator,
                          MobilityScenario, compute_trace_stats)
from adadc_v2 import (AdaDCv2Protocol, build_adadc_v2, SchedulePlan,
                       CorrectionConfig, InferenceMode, ProtocolTrace,
                       HybridScheduler, OfflineScheduler,
                       GreedyOnlineScheduler, OptimalOnlineScheduler,
                       _lookup_fid, _FID_STATE_RELAY)
from sd_latent_ext import (RESOLUTION_CONFIGS, compute_comm_cost_table,
                            compute_scale_latency, compute_seed_sync_crossover,
                            print_comm_cost_table, print_scale_latency_table)


def load_model(model_fn, ckpt_path, device):
    model = model_fn()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()
    print(f"  Loaded {ckpt_path} (epoch {ckpt.get('epoch','?')})")
    return model


# ============================================================
# Exp10: Protocol comparison on real traces
# ============================================================

def run_exp10_protocol_comparison(args, device):
    """
    Compare AdaDC v2 scheduling strategies under diverse network traces.

    For each (trace, scheduler, budget) combination:
      - Run protocol simulation to get plan + latency
      - Lookup expected FID from pre-computed tables
      - Record: FID, latency, comm_kb, num_queries, mode_selected

    This experiment does NOT generate actual images (fast, ~1 min).
    It uses pre-computed FID lookup from Exp1-5 results.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 10: Protocol Comparison on Real Network Traces")
    print("=" * 70)

    # Load system profile
    if args.cloud_ckpt and args.edge_ckpt and os.path.exists(args.cloud_ckpt):
        cloud = load_model(build_cloud_model, args.cloud_ckpt, device)
        edge = load_model(build_edge_model, args.edge_ckpt, device)
        latency_profile = profile_system(edge, cloud, device=device)
    else:
        latency_profile = SIMULATED_PROFILES['mobile_small']
        print("  Using simulated latency profile (mobile_small)")

    # Build trace library
    trace_lib = get_trace_library(seed=42)
    print(f"\n  Loaded {len(trace_lib)} network traces")

    # Select representative traces
    selected_traces = {
        '3gpp_4G_static': trace_lib['3gpp_4G_static'],
        '3gpp_4G_pedestrian': trace_lib['3gpp_4G_pedestrian'],
        '3gpp_4G_vehicular': trace_lib['3gpp_4G_vehicular'],
        '3gpp_5G_pedestrian': trace_lib['3gpp_5G_pedestrian'],
        '3gpp_WiFi_static': trace_lib['3gpp_WiFi_static'],
        'markov_urban_commute': trace_lib['markov_urban_commute'],
        'markov_rural_drive': trace_lib['markov_rural_drive'],
        'markov_bandwidth_spike': trace_lib['markov_bandwidth_spike'],
    }

    schedulers = ['offline', 'greedy', 'optimal', 'hybrid']
    budgets_ms = [500, 1000, 2000, 5000, 10000]
    total_steps = 20

    results = {}

    for trace_name, trace in selected_traces.items():
        stats = compute_trace_stats(trace)
        print(f"\n--- Trace: {trace_name} (mean={stats['bw_mean']:.1f}Mbps, "
              f"CV={stats['bw_cv']:.2f}) ---")

        trace_results = {}

        for sched_name in schedulers:
            for budget in budgets_ms:
                sim = TraceReplaySimulator(trace, inference_duration_s=budget/1000)
                protocol = build_adadc_v2(
                    latency_profile=latency_profile,
                    network_sim=sim,
                    scheduler=sched_name,
                    total_steps=total_steps)

                plan = protocol.plan_offline(budget_ms=budget)

                # Estimate latency with trace bandwidth
                bw_samples = [sim.sample_bandwidth(t) for t in np.linspace(0, 1, 20)]
                avg_bw = np.mean(bw_samples)
                conservative_bw = np.percentile(bw_samples, 25)

                rtt = stats['rtt_mean']
                lat = protocol.latency_est.estimate_plan_latency(
                    plan, conservative_bw, rtt)

                key = f"{sched_name}_budget{budget}"
                trace_results[key] = {
                    'scheduler': sched_name,
                    'budget_ms': budget,
                    'plan_label': plan.label,
                    'num_queries': plan.num_queries,
                    'estimated_fid': plan.estimated_fid,
                    'estimated_latency_ms': lat['total_ms'],
                    'within_budget': lat['total_ms'] <= budget,
                    'comm_kb': plan.estimated_comm_kb,
                    'avg_bw_mbps': float(avg_bw),
                    'conservative_bw_mbps': float(conservative_bw),
                    'edge_compute_ms': lat['edge_compute_ms'],
                    'comm_overhead_ms': lat['comm_overhead_ms'],
                }

        results[trace_name] = {
            'trace_stats': stats,
            'results': trace_results,
        }

    # Save
    out_path = os.path.join(args.output_dir, 'exp10_protocol_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary table
    _print_exp10_summary(results, budgets_ms)

    return results


def _print_exp10_summary(results, budgets_ms):
    """Print summary table for Exp10."""
    print("\n" + "=" * 90)
    print("SUMMARY: Best FID by (Trace, Budget)")
    print("=" * 90)
    print(f"{'Trace':>30s}", end='')
    for b in budgets_ms:
        print(f" {'B='+str(b):>12s}", end='')
    print()
    print('-' * (30 + 13 * len(budgets_ms)))

    for trace_name, data in results.items():
        print(f"{trace_name:>30s}", end='')
        for b in budgets_ms:
            # Find best scheduler for this budget
            best_fid = 999
            best_sched = ""
            for key, r in data['results'].items():
                if r['budget_ms'] == b and r['within_budget']:
                    if r['estimated_fid'] < best_fid:
                        best_fid = r['estimated_fid']
                        best_sched = r['scheduler'][:3]
            if best_fid < 999:
                print(f" {best_fid:>5.1f}({best_sched})", end='')
            else:
                print(f" {'edge':>12s}", end='')
        print()


# ============================================================
# Exp11: Scheduling strategy analysis
# ============================================================

def run_exp11_scheduling_analysis(args, device):
    """
    Compare scheduling strategies in detail:
    - Static: always use same config
    - Periodic: query every k steps regardless
    - Drift-aware: query when drift exceeds threshold
    - Budget-optimal: DP over candidate points

    Uses actual model inference for accurate FID measurement.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 11: Scheduling Strategy Analysis")
    print("=" * 70)

    cloud = load_model(build_cloud_model, args.cloud_ckpt, device)
    edge = load_model(build_edge_model, args.edge_ckpt, device)
    collab = DirectionCorrectionSampler(edge, cloud)
    latency_profile = profile_system(edge, cloud, device=device)

    ref_dir = prepare_cifar10_reference()
    shape = (3, 32, 32)
    total_steps = 20

    # Strategies to compare
    strategies = {
        'edge_only': {
            'corrections': [],
            'description': 'No communication',
        },
        'static_1pt_kr10': {
            'corrections': [CorrectionConfig(0.7, 0.1, 4)],
            'description': 'Static: 1 query at t=0.7, kr=10%',
        },
        'static_3pt_kr20': {
            'corrections': [CorrectionConfig(t, 0.2, 4) for t in [0.3, 0.6, 0.8]],
            'description': 'Static: 3 queries, kr=20%',
        },
        'periodic_5step': {
            'corrections': [CorrectionConfig(t, 0.2, 4) for t in [0.25, 0.5, 0.75]],
            'description': 'Periodic: query every 5 steps, kr=20%',
        },
        'late_heavy': {
            'corrections': [CorrectionConfig(t, 0.5, 4) for t in [0.6, 0.8, 0.9]],
            'description': 'Late-heavy: 3 queries in [0.6, 0.9], kr=50%',
        },
        'early_light': {
            'corrections': [CorrectionConfig(t, 0.1, 4) for t in [0.2, 0.3, 0.4]],
            'description': 'Early-light: 3 queries in [0.2, 0.4], kr=10%',
        },
        'state_relay_05': {
            'corrections': [CorrectionConfig(0.5, 1.0, 32, InferenceMode.STATE_RELAY)],
            'description': 'State Relay at t*=0.5',
        },
    }

    results = {}

    for name, strategy in strategies.items():
        print(f"\n--- {name}: {strategy['description']} ---")
        torch.manual_seed(42)

        corrections = strategy['corrections']
        query_points = [c.t_query for c in corrections
                        if c.mode == InferenceMode.DIRECTION_CORRECTION]
        sr_points = [c.t_query for c in corrections
                     if c.mode == InferenceMode.STATE_RELAY]

        if not corrections:
            # Edge only
            sampler_fn = lambda x0: collab.sample_edge_only(x0, num_steps=total_steps)
        elif sr_points:
            # State relay
            t_star = sr_points[0]
            edge_steps = int(total_steps * t_star)
            cloud_steps = total_steps - edge_steps
            sampler_fn = lambda x0, ts=t_star, es=edge_steps, cs=cloud_steps: \
                collab.sample_state_relay(x0, t_star=ts, edge_steps=es, cloud_steps=cs)
        elif len(query_points) == 1:
            # Single-point DC
            kr = corrections[0].keep_ratio
            cfn = make_compress_fn(kr, 4)
            t_star = query_points[0]
            es_before = max(1, int(total_steps * t_star))
            es_after = total_steps - es_before
            sampler_fn = lambda x0, cf=cfn, ts=t_star, eb=es_before, ea=es_after: \
                collab.sample_direction_correction(
                    x0, t_star=ts, edge_steps_before=eb,
                    edge_steps_after=ea, compress_fn=cf)
        else:
            # Multi-point DC
            kr = corrections[0].keep_ratio
            cfn = make_compress_fn(kr, 4)
            sampler_fn = lambda x0, cf=cfn, qp=query_points: \
                collab.sample_multi_point_correction(
                    x0, query_points=qp, total_steps=total_steps, compress_fn=cf)

        # Generate samples
        all_samples = []
        remaining = args.num_samples
        pbar = tqdm(total=args.num_samples, desc=name)
        while remaining > 0:
            bs = min(args.batch_size, remaining)
            x0 = torch.randn(bs, *shape, device=device)
            out = sampler_fn(x0)
            if isinstance(out, dict):
                out = out['x_final']
            all_samples.append(out.cpu())
            remaining -= bs
            pbar.update(bs)
        pbar.close()
        samples = torch.cat(all_samples, dim=0)[:args.num_samples]

        gen_dir = os.path.join(args.output_dir, 'exp11', name)
        save_samples_to_dir(samples, gen_dir)
        fid = compute_fid(gen_dir, ref_dir, device=str(device))

        # Communication cost
        total_comm_kb = sum(c.dv_size_kb for c in corrections
                            if c.mode == InferenceMode.DIRECTION_CORRECTION)
        if sr_points:
            total_comm_kb = 2 * 3 * 32 * 32 * 4 / 1024  # state relay: full state × 2

        results[name] = {
            'description': strategy['description'],
            'fid': fid,
            'num_queries': len(corrections),
            'query_points': [c.t_query for c in corrections],
            'comm_kb': total_comm_kb,
        }
        print(f"  FID={fid:.2f}, queries={len(corrections)}, comm={total_comm_kb:.2f}KB")

    # Save
    out_path = os.path.join(args.output_dir, 'exp11_scheduling.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 70)
    print(f"{'Strategy':>25s} {'FID':>8s} {'Queries':>8s} {'Comm(KB)':>10s}")
    print('-' * 55)
    for name, r in sorted(results.items(), key=lambda x: x[1]['fid']):
        print(f"{name:>25s} {r['fid']:>8.2f} {r['num_queries']:>8d} {r['comm_kb']:>10.2f}")

    return results


# ============================================================
# Exp12: Scalability across resolutions
# ============================================================

def run_exp12_scalability(args, device):
    """
    Scalability analysis: communication + latency across resolution scales.
    No actual model training needed — uses analytical computation.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 12: Scalability Analysis")
    print("=" * 70)

    # Communication cost table
    print("\n--- Communication Cost Table ---")
    comm_table = compute_comm_cost_table()
    print_comm_cost_table(comm_table)

    # Latency comparison
    print("\n--- Latency Comparison ---")
    latency_results = compute_scale_latency()
    print_scale_latency_table(latency_results)

    # Seed-sync crossover
    print("\n--- Seed-Sync Crossover ---")
    crossover = compute_seed_sync_crossover()

    # Save all
    results = {
        'comm_table': {},
        'latency': [],
        'seed_sync_crossover': crossover,
    }

    for res_name, entries in comm_table.items():
        results['comm_table'][res_name] = [
            {'method': e.method, 'up_kb': e.uplink_kb,
             'down_kb': e.downlink_kb, 'total_kb': e.total_kb}
            for e in entries
        ]

    for r in latency_results:
        results['latency'].append({
            'resolution': r.resolution,
            'network': r.network,
            'edge_only_ms': r.edge_only_ms,
            'state_relay_ms': r.state_relay_ms,
            'dc_naive_ms': r.dc_naive_ms,
            'dc_seedsync_ms': r.dc_seedsync_ms,
            'dc_3pt_seedsync_ms': r.dc_3pt_seedsync_ms,
            'cloud_only_ms': r.cloud_only_ms,
            'dc_speedup_vs_sr': r.dc_speedup_vs_sr,
        })

    out_path = os.path.join(args.output_dir, 'exp12_scalability.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return results


# ============================================================
# Exp13: Seed-sync ablation
# ============================================================

def run_exp13_seedsync(args, device):
    """
    Ablation: naive upload vs seed-sync protocol.

    Measures the actual latency impact of avoiding x_t* upload.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 13: Seed-Sync Ablation")
    print("=" * 70)

    if args.cloud_ckpt and os.path.exists(args.cloud_ckpt):
        cloud = load_model(build_cloud_model, args.cloud_ckpt, device)
        edge = load_model(build_edge_model, args.edge_ckpt, device)
        latency_profile = profile_system(edge, cloud, device=device)
    else:
        latency_profile = SIMULATED_PROFILES['mobile_small']

    results = []

    for res_name, cfg in RESOLUTION_CONFIGS.items():
        state_kb = cfg.state_size_kb
        dv_kb = compute_transmitted_size_kb(cfg.data_shape, 0.1, 4)
        seed_kb = 0.032

        # Cloud running edge model (roughly 10x faster than mobile edge)
        edge_on_cloud_ms = cfg.edge_params_M * 0.02  # ms per step on cloud
        t_star = 0.5
        recon_steps = int(cfg.typical_steps * t_star)
        recon_ms = recon_steps * edge_on_cloud_ms

        for net in [NetworkProfile.G3, NetworkProfile.G4, NetworkProfile.WIFI, NetworkProfile.G5]:
            params = NETWORK_PARAMS[net]
            bw_bps = max(params.bw_mean_mbps * 1e6, 1.0)
            rtt_half = params.rtt_ms / 2

            # Naive: upload x_t* + download δv
            upload_ms = (state_kb * 8 * 1024 / bw_bps) * 1000 + rtt_half
            download_ms = (dv_kb * 8 * 1024 / bw_bps) * 1000 + rtt_half
            naive_comm_ms = upload_ms + download_ms

            # Seed-sync: upload seed + cloud reconstructs + download δv
            seed_upload_ms = (seed_kb * 8 * 1024 / bw_bps) * 1000 + rtt_half
            seedsync_comm_ms = seed_upload_ms + recon_ms + download_ms

            improvement_ms = naive_comm_ms - seedsync_comm_ms

            results.append({
                'resolution': res_name,
                'network': net.value,
                'state_kb': state_kb,
                'dv_kb': dv_kb,
                'naive_upload_ms': upload_ms,
                'seed_upload_ms': seed_upload_ms,
                'recon_ms': recon_ms,
                'naive_total_ms': naive_comm_ms,
                'seedsync_total_ms': seedsync_comm_ms,
                'improvement_ms': improvement_ms,
                'seedsync_faster': improvement_ms > 0,
            })

    # Print summary
    print(f"\n{'Resolution':>20s} {'Network':>8s} {'State(KB)':>10s} "
          f"{'Naive(ms)':>10s} {'SeedSync(ms)':>12s} {'Saved(ms)':>10s} {'Winner':>10s}")
    print('-' * 90)
    for r in results:
        winner = 'SeedSync' if r['seedsync_faster'] else 'Naive'
        print(f"{r['resolution']:>20s} {r['network']:>8s} {r['state_kb']:>10.1f} "
              f"{r['naive_total_ms']:>10.1f} {r['seedsync_total_ms']:>12.1f} "
              f"{r['improvement_ms']:>10.1f} {winner:>10s}")

    out_path = os.path.join(args.output_dir, 'exp13_seedsync.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    return results


# ============================================================
# Exp14: Dynamic bandwidth stress test
# ============================================================

def run_exp14_dynamic_bandwidth(args, device):
    """
    Stress test: protocol behavior under bandwidth transitions.

    Scenarios:
    - Stable 4G → sudden drop to 3G
    - 3G → sudden spike to WiFi
    - Random Markov switching
    - Gradual degradation

    Shows how Hybrid scheduler adapts vs static strategies.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 14: Dynamic Bandwidth Stress Test")
    print("=" * 70)

    latency_profile = SIMULATED_PROFILES['mobile_small']

    # Build stress scenarios
    scenarios = {}

    # 1. Sudden drop
    scenarios['sudden_drop'] = generate_markov_trace(
        states={'good_4G': (15.0, 3.0, 40.0), 'bad_3G': (1.0, 0.5, 120.0)},
        transition_matrix=np.array([[0.98, 0.02], [0.01, 0.99]]),
        duration_s=10.0, state_duration_s=3.0, seed=42)

    # 2. Sudden spike
    scenarios['sudden_spike'] = generate_markov_trace(
        states={'bad_3G': (1.0, 0.5, 120.0), 'good_WiFi': (50.0, 10.0, 8.0)},
        transition_matrix=np.array([[0.98, 0.02], [0.01, 0.99]]),
        duration_s=10.0, state_duration_s=3.0, seed=42)

    # 3. Rapid fluctuation
    scenarios['rapid_fluctuation'] = generate_markov_trace(
        states={'low': (2.0, 1.0, 80.0), 'mid': (10.0, 4.0, 40.0),
                'high': (40.0, 10.0, 10.0)},
        transition_matrix=np.array([
            [0.7, 0.2, 0.1],
            [0.15, 0.7, 0.15],
            [0.1, 0.2, 0.7]]),
        duration_s=10.0, state_duration_s=1.0, seed=42)

    # 4. Gradual degradation (build manually)
    scenarios['gradual_degrade'] = generate_3gpp_trace(
        '4G', MobilityScenario.VEHICULAR, duration_s=10.0, seed=42)

    schedulers = ['offline', 'greedy', 'optimal', 'hybrid']
    budgets = [1000, 2000, 5000]

    results = {}

    for scenario_name, trace in scenarios.items():
        stats = compute_trace_stats(trace)
        print(f"\n--- Scenario: {scenario_name} "
              f"(mean_bw={stats['bw_mean']:.1f}, CV={stats['bw_cv']:.2f}) ---")

        scenario_results = {}

        for sched in schedulers:
            for budget in budgets:
                sim = TraceReplaySimulator(trace, inference_duration_s=budget/1000)
                protocol = build_adadc_v2(
                    latency_profile=latency_profile,
                    network_sim=sim,
                    scheduler=sched,
                    total_steps=20)

                # Run multiple simulations with different trace offsets
                offsets = np.linspace(0, max(trace.duration_s - budget/1000, 0.1),
                                      min(20, int(trace.duration_s)))
                fids = []
                latencies = []
                comms = []

                for offset in offsets:
                    sim.trace_offset_s = offset
                    plan = protocol.plan_offline(budget_ms=budget)
                    bw_samples = [sim.sample_bandwidth(t) for t in np.linspace(0, 1, 20)]
                    avg_bw = np.percentile(bw_samples, 25)
                    rtt = stats['rtt_mean']
                    lat = protocol.latency_est.estimate_plan_latency(plan, avg_bw, rtt)

                    fids.append(plan.estimated_fid)
                    latencies.append(lat['total_ms'])
                    comms.append(plan.estimated_comm_kb)

                key = f"{sched}_b{budget}"
                scenario_results[key] = {
                    'scheduler': sched,
                    'budget_ms': budget,
                    'fid_mean': float(np.mean(fids)),
                    'fid_std': float(np.std(fids)),
                    'latency_mean_ms': float(np.mean(latencies)),
                    'latency_p95_ms': float(np.percentile(latencies, 95)),
                    'comm_mean_kb': float(np.mean(comms)),
                    'budget_violations': float(np.mean([1 for l in latencies if l > budget])),
                }
                print(f"  {sched:>8s} B={budget:>5d}: FID={np.mean(fids):.1f}±{np.std(fids):.1f}, "
                      f"lat={np.mean(latencies):.0f}ms, "
                      f"violations={np.mean([1 for l in latencies if l > budget]):.0%}")

        results[scenario_name] = {
            'trace_stats': stats,
            'results': scenario_results,
        }

    out_path = os.path.join(args.output_dir, 'exp14_dynamic_bw.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    return results


# ============================================================
# Run all
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='System experiments v2 (INFOCOM)')
    parser.add_argument('--cloud_ckpt', type=str,
                        default='checkpoints/cloud_best.pt')
    parser.add_argument('--edge_ckpt', type=str,
                        default='checkpoints/edge_best.pt')
    parser.add_argument('--output_dir', type=str, default='results/v2/')
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')

    # Experiment selection
    parser.add_argument('--exp10', action='store_true', help='Protocol comparison')
    parser.add_argument('--exp11', action='store_true', help='Scheduling analysis')
    parser.add_argument('--exp12', action='store_true', help='Scalability')
    parser.add_argument('--exp13', action='store_true', help='Seed-sync ablation')
    parser.add_argument('--exp14', action='store_true', help='Dynamic bandwidth')
    parser.add_argument('--all', action='store_true', help='Run all')

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    if args.exp10 or args.all:
        run_exp10_protocol_comparison(args, device)
    if args.exp11 or args.all:
        run_exp11_scheduling_analysis(args, device)
    if args.exp12 or args.all:
        run_exp12_scalability(args, device)
    if args.exp13 or args.all:
        run_exp13_seedsync(args, device)
    if args.exp14 or args.all:
        run_exp14_dynamic_bandwidth(args, device)

    if not any([args.exp10, args.exp11, args.exp12, args.exp13, args.exp14, args.all]):
        print("Usage: python experiment_v2.py --exp10|--exp11|--exp12|--exp13|--exp14|--all")
        print("  --exp10: Protocol comparison on real traces (fast, no FID)")
        print("  --exp11: Scheduling strategy analysis (slow, generates samples)")
        print("  --exp12: Scalability across resolutions (fast, analytical)")
        print("  --exp13: Seed-sync ablation (fast, analytical)")
        print("  --exp14: Dynamic bandwidth stress test (fast, simulation)")
        print("  --all:   Run all experiments")


if __name__ == '__main__':
    main()
