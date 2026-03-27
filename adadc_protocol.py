"""
AdaDC: Adaptive Direction Correction Protocol.

Core system contribution: an online protocol that selects optimal
(num_query_points, compression_config) given real-time network bandwidth
to maximize generation quality under a latency deadline.
"""

import json
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional

from network_model import NetworkSimulator, NetworkProfile
from latency_profiler import SystemLatencyProfile
from compression import compute_compression_ratio, make_compress_fn


@dataclass
class AdaDCConfig:
    """A specific direction correction configuration."""
    num_queries: int              # 0=edge-only, 1=single-point, 2/3=multi-point
    query_points: list            # t values to query cloud
    keep_ratio: float = 1.0      # sparsification ratio
    bits: int = 4                 # quantization bits
    fid: float = 0.0             # expected FID (from lookup)
    comm_kb: float = 0.0         # total communication cost (KB)
    label: str = ""

    def __repr__(self):
        return (f"AdaDCConfig(q={self.num_queries}, pts={self.query_points}, "
                f"kr={self.keep_ratio}, bits={self.bits}, "
                f"FID={self.fid:.1f}, comm={self.comm_kb:.2f}KB)")


class ParetoFrontier:
    """
    Builds and queries the Pareto frontier of FID vs communication cost
    from pre-computed experimental results.

    A configuration is Pareto-optimal if no other configuration achieves
    both lower FID and lower communication cost.
    """

    def __init__(self):
        self.all_configs: list[AdaDCConfig] = []
        self.pareto_configs: list[AdaDCConfig] = []

    def build_from_results(self, results_dir: str = 'results/') -> 'ParetoFrontier':
        """Load experiment results and build Pareto frontier."""
        # Load Exp1: baseline results
        try:
            with open(f'{results_dir}/experiment1_results.json') as f:
                exp1 = json.load(f)
            fid_scores = exp1['fid_scores']

            # Edge-only: zero communication
            self.all_configs.append(AdaDCConfig(
                num_queries=0, query_points=[], keep_ratio=0, bits=0,
                fid=fid_scores.get('Edge Only', 65.0), comm_kb=0.0,
                label='Edge Only',
            ))
        except (FileNotFoundError, KeyError):
            # Fallback defaults
            self.all_configs.append(AdaDCConfig(
                num_queries=0, query_points=[], keep_ratio=0, bits=0,
                fid=63.0, comm_kb=0.0, label='Edge Only',
            ))

        # Load Exp3: single-point compression sweep
        try:
            with open(f'{results_dir}/experiment3_compression_sweep.json') as f:
                exp3 = json.load(f)
            for entry in exp3:
                self.all_configs.append(AdaDCConfig(
                    num_queries=1,
                    query_points=[0.7],
                    keep_ratio=entry['keep_ratio'],
                    bits=entry['bits'],
                    fid=entry['fid'],
                    comm_kb=entry['compressed_kb'],
                    label=f"1pt-kr{entry['keep_ratio']}-b{entry['bits']}",
                ))
        except FileNotFoundError:
            pass

        # Load Exp4: multi-point (uncompressed)
        try:
            with open(f'{results_dir}/experiment4_multi_point.json') as f:
                exp4 = json.load(f)
            for entry in exp4:
                if entry['num_queries'] > 0:
                    self.all_configs.append(AdaDCConfig(
                        num_queries=entry['num_queries'],
                        query_points=entry['query_points'],
                        keep_ratio=1.0,
                        bits=32,
                        fid=entry['fid'],
                        comm_kb=entry['comm_kb'],
                        label=entry['label'],
                    ))
        except FileNotFoundError:
            pass

        # Load Exp5: multi-point + compression
        try:
            with open(f'{results_dir}/experiment5_multi_point_compression.json') as f:
                exp5 = json.load(f)
            for entry in exp5:
                self.all_configs.append(AdaDCConfig(
                    num_queries=entry['num_queries'],
                    query_points=entry['query_points'],
                    keep_ratio=entry['keep_ratio'],
                    bits=entry['bits'],
                    fid=entry['fid'],
                    comm_kb=entry['total_comm_kb'],
                    label=entry['label'] + f"-kr{entry['keep_ratio']}-b{entry['bits']}",
                ))
        except FileNotFoundError:
            pass

        self._compute_pareto()
        return self

    def _compute_pareto(self):
        """Extract Pareto-optimal configurations (lower FID, lower comm)."""
        # Sort by comm_kb ascending
        sorted_configs = sorted(self.all_configs, key=lambda c: c.comm_kb)

        pareto = []
        best_fid = float('inf')
        for cfg in sorted_configs:
            if cfg.fid < best_fid:
                pareto.append(cfg)
                best_fid = cfg.fid

        self.pareto_configs = pareto

    def get_best_config(self, max_comm_kb: float) -> AdaDCConfig:
        """
        Get the best (lowest FID) Pareto-optimal config within comm budget.

        Args:
            max_comm_kb: maximum allowed communication in KB
        Returns:
            Best AdaDCConfig, or edge-only if nothing fits
        """
        feasible = [c for c in self.pareto_configs if c.comm_kb <= max_comm_kb]
        if not feasible:
            # Fall back to edge-only
            return self.all_configs[0] if self.all_configs else AdaDCConfig(
                num_queries=0, query_points=[], fid=65.0, comm_kb=0.0)
        # Return the one with lowest FID among feasible
        return min(feasible, key=lambda c: c.fid)

    def get_pareto_points(self) -> list:
        """Get all Pareto-optimal points as (fid, comm_kb, config) tuples."""
        return [(c.fid, c.comm_kb, c) for c in self.pareto_configs]

    def get_all_points(self) -> list:
        """Get all data points (including non-Pareto) for plotting."""
        return [(c.fid, c.comm_kb, c) for c in self.all_configs]


@dataclass
class ProtocolTrace:
    """Records all decisions and timings during adaptive generation."""
    decisions: list = field(default_factory=list)
    total_edge_compute_ms: float = 0.0
    total_cloud_compute_ms: float = 0.0
    total_uplink_ms: float = 0.0
    total_downlink_ms: float = 0.0
    total_comm_kb: float = 0.0
    selected_config: Optional[AdaDCConfig] = None

    @property
    def total_latency_ms(self) -> float:
        return (self.total_edge_compute_ms + self.total_cloud_compute_ms
                + self.total_uplink_ms + self.total_downlink_ms)

    def to_dict(self) -> dict:
        return {
            'decisions': self.decisions,
            'total_edge_compute_ms': self.total_edge_compute_ms,
            'total_cloud_compute_ms': self.total_cloud_compute_ms,
            'total_uplink_ms': self.total_uplink_ms,
            'total_downlink_ms': self.total_downlink_ms,
            'total_comm_kb': self.total_comm_kb,
            'total_latency_ms': self.total_latency_ms,
            'selected_config': str(self.selected_config),
        }


class AdaDCProtocol:
    """
    Adaptive Direction Correction Protocol.

    Given a latency budget and real-time network conditions, selects the
    optimal correction strategy (num queries, compression level) to
    maximize generation quality.

    Two operating modes:
    1. Offline: select strategy before generation based on expected bandwidth
    2. Online: adapt at each potential query point during generation
    """

    # Candidate query point sets
    CANDIDATE_QUERY_POINTS = [
        [],                    # edge-only
        [0.7],                 # 1-point
        [0.5, 0.8],           # 2-point
        [0.3, 0.6, 0.8],     # 3-point
    ]

    CANDIDATE_KEEP_RATIOS = [0.05, 0.1, 0.2, 0.5, 1.0]
    CANDIDATE_BITS = [4]  # 4-bit sufficient per exp3 results

    def __init__(self, pareto: ParetoFrontier,
                 latency_profile: SystemLatencyProfile,
                 network_sim: NetworkSimulator):
        self.pareto = pareto
        self.latency = latency_profile
        self.network = network_sim

    def estimate_total_latency(self, config: AdaDCConfig,
                                total_steps: int = 20,
                                bandwidth_mbps: float = None,
                                t_query: float = 0.5) -> dict:
        """
        Estimate end-to-end latency for a given configuration.

        Latency breakdown:
        - edge_compute: total_steps * edge_step_ms
        - cloud_compute: num_queries * cloud_query_ms
        - uplink: num_queries * transmission_time(delta_v_request)
        - downlink: num_queries * transmission_time(delta_v_response)

        For DC, uplink sends x_t* (for cloud to compute v_cloud),
        downlink receives compressed delta_v.
        """
        edge_ms = total_steps * self.latency.edge_step_ms
        cloud_ms = config.num_queries * self.latency.cloud_query_ms

        if bandwidth_mbps is None:
            bandwidth_mbps = self.network.sample_bandwidth(t_query)

        bw_bps = bandwidth_mbps * 1e6

        # For each query: edge sends x_t* (full state), cloud returns δv (compressed)
        state_size_kb = 3 * 32 * 32 * 32 / 8 / 1024  # 12 KB for CIFAR-10
        dv_size_kb = config.comm_kb / max(config.num_queries, 1)

        rtt_half = self.network.params.rtt_ms / 2

        uplink_per_query = (state_size_kb * 8 * 1024 / bw_bps) * 1000 + rtt_half
        downlink_per_query = (dv_size_kb * 8 * 1024 / bw_bps) * 1000 + rtt_half

        uplink_ms = config.num_queries * uplink_per_query
        downlink_ms = config.num_queries * downlink_per_query

        return {
            'edge_compute_ms': edge_ms,
            'cloud_compute_ms': cloud_ms,
            'uplink_ms': uplink_ms,
            'downlink_ms': downlink_ms,
            'total_ms': edge_ms + cloud_ms + uplink_ms + downlink_ms,
            'bandwidth_mbps': bandwidth_mbps,
        }

    def select_strategy_offline(self, latency_budget_ms: float,
                                 total_steps: int = 20) -> AdaDCConfig:
        """
        Offline strategy selection: pick the best config given expected bandwidth.

        Iterates over all Pareto-optimal configs and selects the one with
        lowest FID that fits within the latency budget.
        """
        avg_bw = np.mean([self.network.sample_bandwidth(t)
                          for t in np.linspace(0, 1, 20)])

        best_config = None
        best_fid = float('inf')

        for config in self.pareto.pareto_configs:
            lat = self.estimate_total_latency(
                config, total_steps, bandwidth_mbps=avg_bw)
            if lat['total_ms'] <= latency_budget_ms and config.fid < best_fid:
                best_fid = config.fid
                best_config = config

        if best_config is None:
            # Nothing fits: use edge-only
            best_config = AdaDCConfig(
                num_queries=0, query_points=[], fid=63.0, comm_kb=0.0,
                label='Edge Only (fallback)')

        return best_config

    def select_strategy_online(self, latency_budget_ms: float,
                                elapsed_ms: float,
                                current_t: float,
                                total_steps: int = 20) -> AdaDCConfig:
        """
        Online strategy selection: adapt based on current bandwidth and
        remaining latency budget.

        Called at each candidate query point during generation.
        """
        remaining_budget = latency_budget_ms - elapsed_ms
        if remaining_budget <= 0:
            return AdaDCConfig(num_queries=0, query_points=[],
                               fid=63.0, comm_kb=0.0, label='Edge Only (timeout)')

        current_bw = self.network.sample_bandwidth(current_t)
        remaining_steps = max(1, int(total_steps * (1 - current_t)))

        best_config = None
        best_fid = float('inf')

        # Try single-point correction at current_t with various compressions
        for kr in self.CANDIDATE_KEEP_RATIOS:
            for bits in self.CANDIDATE_BITS:
                cfg = AdaDCConfig(
                    num_queries=1,
                    query_points=[current_t],
                    keep_ratio=kr,
                    bits=bits,
                )
                # Estimate comm size
                total_elements = 3 * 32 * 32
                kept = int(total_elements * kr)
                cfg.comm_kb = (kept * (bits + 16) + 64) / 8 / 1024

                # Look up expected FID from Pareto (approximate)
                closest = self.pareto.get_best_config(cfg.comm_kb)
                cfg.fid = closest.fid

                lat = self.estimate_total_latency(
                    cfg, remaining_steps, bandwidth_mbps=current_bw)

                # Check if communication fits in remaining budget
                comm_latency = lat['cloud_compute_ms'] + lat['uplink_ms'] + lat['downlink_ms']
                edge_remaining = remaining_steps * self.latency.edge_step_ms

                if comm_latency + edge_remaining <= remaining_budget:
                    if cfg.fid < best_fid:
                        best_fid = cfg.fid
                        best_config = cfg

        if best_config is None:
            return AdaDCConfig(num_queries=0, query_points=[],
                               fid=63.0, comm_kb=0.0, label='Skip (no budget)')

        return best_config

    def simulate_generation(self, total_steps: int = 20,
                             latency_budget_ms: float = 5000.0,
                             mode: str = 'offline') -> ProtocolTrace:
        """
        Simulate a complete generation run and record protocol decisions.

        Args:
            total_steps: total Euler steps
            latency_budget_ms: end-to-end latency deadline
            mode: 'offline' (select once) or 'online' (adapt at each point)
        Returns:
            ProtocolTrace with all decisions and timing
        """
        trace = ProtocolTrace()

        if mode == 'offline':
            config = self.select_strategy_offline(latency_budget_ms, total_steps)
            trace.selected_config = config

            lat = self.estimate_total_latency(config, total_steps)
            trace.total_edge_compute_ms = lat['edge_compute_ms']
            trace.total_cloud_compute_ms = lat['cloud_compute_ms']
            trace.total_uplink_ms = lat['uplink_ms']
            trace.total_downlink_ms = lat['downlink_ms']
            trace.total_comm_kb = config.comm_kb

            trace.decisions.append({
                'mode': 'offline',
                'config': str(config),
                'latency': lat,
            })

        elif mode == 'online':
            # Simulate step-by-step with online decisions
            candidate_ts = [0.3, 0.5, 0.7, 0.8]
            elapsed_ms = 0.0
            queries_made = 0
            total_comm = 0.0

            dt_steps = 1.0 / total_steps

            for step in range(total_steps):
                current_t = step * dt_steps

                # Edge compute for this step
                elapsed_ms += self.latency.edge_step_ms

                # Check if this is a candidate query point
                for cand_t in candidate_ts:
                    if abs(current_t - cand_t) < dt_steps / 2:
                        # Decide whether to query
                        config = self.select_strategy_online(
                            latency_budget_ms, elapsed_ms, current_t, total_steps)

                        if config.num_queries > 0:
                            # Execute query
                            bw = self.network.sample_bandwidth(current_t)
                            lat = self.estimate_total_latency(
                                config, 1, bandwidth_mbps=bw)
                            elapsed_ms += (lat['cloud_compute_ms']
                                           + lat['uplink_ms']
                                           + lat['downlink_ms'])
                            total_comm += config.comm_kb
                            queries_made += 1

                            trace.decisions.append({
                                'step': step,
                                't': current_t,
                                'action': 'query',
                                'bandwidth_mbps': bw,
                                'config': str(config),
                                'comm_kb': config.comm_kb,
                                'elapsed_ms': elapsed_ms,
                            })
                        else:
                            trace.decisions.append({
                                'step': step,
                                't': current_t,
                                'action': 'skip',
                                'reason': config.label,
                                'elapsed_ms': elapsed_ms,
                            })
                        break

            trace.total_edge_compute_ms = total_steps * self.latency.edge_step_ms
            trace.total_cloud_compute_ms = queries_made * self.latency.cloud_query_ms
            trace.total_comm_kb = total_comm

        return trace

    def compute_method_latency(self, method: str,
                                total_steps: int = 20,
                                t_star: float = 0.7,
                                config: AdaDCConfig = None) -> dict:
        """
        Compute latency breakdown for a specific method.

        Methods: 'edge_only', 'cloud_only', 'state_relay', 'dc', 'adadc'
        """
        avg_bw = np.mean([self.network.sample_bandwidth(t)
                          for t in np.linspace(0, 1, 20)])
        bw_bps = avg_bw * 1e6
        rtt_half = self.network.params.rtt_ms / 2
        state_kb = 3 * 32 * 32 * 32 / 8 / 1024  # 12 KB

        if method == 'edge_only':
            return {
                'method': 'Edge Only',
                'edge_compute_ms': total_steps * self.latency.edge_step_ms,
                'cloud_compute_ms': 0,
                'uplink_ms': 0, 'downlink_ms': 0,
                'total_ms': total_steps * self.latency.edge_step_ms,
                'comm_kb': 0,
            }

        elif method == 'cloud_only':
            # Full offloading: send noise, cloud generates, send back image
            up_ms = (state_kb * 8 * 1024 / bw_bps) * 1000 + rtt_half
            cloud_ms = total_steps * self.latency.cloud_step_ms
            down_ms = (state_kb * 8 * 1024 / bw_bps) * 1000 + rtt_half
            return {
                'method': 'Cloud Only',
                'edge_compute_ms': 0,
                'cloud_compute_ms': cloud_ms,
                'uplink_ms': up_ms, 'downlink_ms': down_ms,
                'total_ms': up_ms + cloud_ms + down_ms,
                'comm_kb': 2 * state_kb,
            }

        elif method == 'state_relay':
            steps_before = max(1, int(total_steps * t_star))
            steps_after = total_steps - steps_before
            edge_ms = steps_before * self.latency.edge_step_ms
            # Send full state to cloud
            up_ms = (state_kb * 8 * 1024 / bw_bps) * 1000 + rtt_half
            cloud_ms = steps_after * self.latency.cloud_step_ms
            # Cloud sends back final image
            down_ms = (state_kb * 8 * 1024 / bw_bps) * 1000 + rtt_half
            return {
                'method': 'State Relay',
                'edge_compute_ms': edge_ms,
                'cloud_compute_ms': cloud_ms,
                'uplink_ms': up_ms, 'downlink_ms': down_ms,
                'total_ms': edge_ms + up_ms + cloud_ms + down_ms,
                'comm_kb': 2 * state_kb,
            }

        elif method == 'dc':
            if config is None:
                config = self.pareto.get_best_config(max_comm_kb=12.0)
            return self.estimate_total_latency(config, total_steps)

        elif method == 'adadc':
            if config is None:
                config = self.select_strategy_offline(
                    latency_budget_ms=float('inf'), total_steps=total_steps)
            lat = self.estimate_total_latency(config, total_steps)
            lat['method'] = f'AdaDC ({config.label})'
            lat['comm_kb'] = config.comm_kb
            return lat

        else:
            raise ValueError(f"Unknown method: {method}")


def build_protocol(results_dir: str = 'results/',
                   latency_profile: SystemLatencyProfile = None,
                   network_profile: NetworkProfile = NetworkProfile.G4,
                   seed: int = 42) -> AdaDCProtocol:
    """
    Convenience function to build a complete AdaDC protocol instance.

    Args:
        results_dir: path to experiment results
        latency_profile: measured or simulated system latency
        network_profile: network condition
        seed: random seed
    Returns:
        Configured AdaDCProtocol
    """
    pareto = ParetoFrontier().build_from_results(results_dir)
    network = NetworkSimulator(network_profile, seed=seed)

    if latency_profile is None:
        from latency_profiler import SIMULATED_PROFILES
        latency_profile = SIMULATED_PROFILES['mobile_small']

    return AdaDCProtocol(pareto, latency_profile, network)
