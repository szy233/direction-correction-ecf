"""
real_traces.py — Real-world network trace loading, replay, and generation.

Supports:
1. Synthetic traces based on 3GPP channel models (TR 38.901)
2. Loading external CSV/JSON trace files (FCC MBA, Oboe, etc.)
3. Markov-modulated bandwidth traces for dynamic scenarios

Used by AdaDC v2 protocol for realistic end-to-end evaluation.
"""

import os
import json
import csv
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from network_model import NetworkParams, NETWORK_PARAMS, NetworkProfile


# ============================================================
# Trace data structures
# ============================================================

@dataclass
class TracePoint:
    """Single measurement in a network trace."""
    time_s: float       # wall-clock time (seconds)
    bw_mbps: float      # throughput (Mbps)
    rtt_ms: float       # RTT (ms)
    loss_rate: float    # packet loss rate [0,1]


@dataclass
class NetworkTrace:
    """A complete bandwidth trace with metadata."""
    name: str
    points: list                 # list of TracePoint
    duration_s: float            # total trace duration
    mean_bw_mbps: float
    std_bw_mbps: float
    source: str = "synthetic"    # "synthetic", "fcc", "oboe", "3gpp", "custom"

    @property
    def num_points(self):
        return len(self.points)

    def bw_at_time(self, t_s: float) -> float:
        """Interpolate bandwidth at arbitrary time via nearest neighbor."""
        if not self.points:
            return 0.0
        if t_s <= self.points[0].time_s:
            return self.points[0].bw_mbps
        if t_s >= self.points[-1].time_s:
            return self.points[-1].bw_mbps
        # Binary search
        lo, hi = 0, len(self.points) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self.points[mid].time_s <= t_s:
                lo = mid
            else:
                hi = mid
        # Linear interpolation
        p0, p1 = self.points[lo], self.points[hi]
        alpha = (t_s - p0.time_s) / max(p1.time_s - p0.time_s, 1e-6)
        return p0.bw_mbps + alpha * (p1.bw_mbps - p0.bw_mbps)

    def rtt_at_time(self, t_s: float) -> float:
        """Interpolate RTT at arbitrary time."""
        if not self.points:
            return 50.0
        if t_s <= self.points[0].time_s:
            return self.points[0].rtt_ms
        if t_s >= self.points[-1].time_s:
            return self.points[-1].rtt_ms
        lo, hi = 0, len(self.points) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self.points[mid].time_s <= t_s:
                lo = mid
            else:
                hi = mid
        p0, p1 = self.points[lo], self.points[hi]
        alpha = (t_s - p0.time_s) / max(p1.time_s - p0.time_s, 1e-6)
        return p0.rtt_ms + alpha * (p1.rtt_ms - p0.rtt_ms)

    def slice(self, start_s: float, end_s: float) -> 'NetworkTrace':
        """Extract a sub-trace within [start_s, end_s]."""
        pts = [p for p in self.points if start_s <= p.time_s <= end_s]
        bws = [p.bw_mbps for p in pts] if pts else [0.0]
        return NetworkTrace(
            name=f"{self.name}[{start_s:.0f}-{end_s:.0f}s]",
            points=pts,
            duration_s=end_s - start_s,
            mean_bw_mbps=float(np.mean(bws)),
            std_bw_mbps=float(np.std(bws)),
            source=self.source,
        )

    def to_normalized(self, total_duration_s: float = None) -> 'NetworkTrace':
        """Normalize time axis to [0, total_duration_s]."""
        if not self.points:
            return self
        dur = total_duration_s or self.duration_s
        t0 = self.points[0].time_s
        t_range = self.points[-1].time_s - t0
        if t_range < 1e-6:
            return self
        new_points = []
        for p in self.points:
            new_t = (p.time_s - t0) / t_range * dur
            new_points.append(TracePoint(new_t, p.bw_mbps, p.rtt_ms, p.loss_rate))
        bws = [p.bw_mbps for p in new_points]
        return NetworkTrace(
            name=self.name, points=new_points, duration_s=dur,
            mean_bw_mbps=float(np.mean(bws)),
            std_bw_mbps=float(np.std(bws)),
            source=self.source,
        )


# ============================================================
# Trace-driven network simulator (drop-in for NetworkSimulator)
# ============================================================

class TraceReplaySimulator:
    """
    Replays a real network trace during generation.
    Drop-in replacement for NetworkSimulator — same interface.

    Maps normalized generation time t in [0,1] to wall-clock time
    in the trace, accounting for the total inference duration.
    """

    def __init__(self, trace: NetworkTrace, inference_duration_s: float = 5.0,
                 trace_offset_s: float = 0.0, seed: int = 42):
        """
        Args:
            trace: NetworkTrace to replay
            inference_duration_s: expected total inference time (seconds)
            trace_offset_s: offset into the trace (for random starts)
            seed: RNG seed for jitter
        """
        self.trace = trace
        self.inference_duration_s = inference_duration_s
        self.trace_offset_s = trace_offset_s
        self.rng = np.random.RandomState(seed)

        # Build params from trace statistics for compatibility
        self.params = NetworkParams(
            bw_mean_mbps=trace.mean_bw_mbps,
            bw_std_mbps=trace.std_bw_mbps,
            bw_min_mbps=min(p.bw_mbps for p in trace.points) if trace.points else 1.0,
            bw_max_mbps=max(p.bw_mbps for p in trace.points) if trace.points else 100.0,
            rtt_ms=float(np.mean([p.rtt_ms for p in trace.points])) if trace.points else 50.0,
            jitter_ms=float(np.std([p.rtt_ms for p in trace.points])) if trace.points else 5.0,
            ar_coeff=0.9,
        )
        self.profile = trace.name

    def sample_bandwidth(self, t: float) -> float:
        """Get bandwidth at normalized time t in [0, 1]."""
        wall_t = self.trace_offset_s + t * self.inference_duration_s
        return self.trace.bw_at_time(wall_t)

    def compute_transmission_time(self, data_size_kb: float, t: float = 0.5) -> float:
        """Compute transmission time at normalized time t."""
        bw_mbps = self.sample_bandwidth(t)
        bw_bps = bw_mbps * 1e6
        data_bits = data_size_kb * 8 * 1024
        transfer_ms = (data_bits / max(bw_bps, 1.0)) * 1000
        rtt = self.trace.rtt_at_time(self.trace_offset_s + t * self.inference_duration_s)
        jitter = abs(self.rng.randn() * self.params.jitter_ms)
        return transfer_ms + rtt / 2 + jitter

    def compute_round_trip_transmission(self, uplink_kb: float,
                                         downlink_kb: float,
                                         t: float = 0.5) -> dict:
        bw_mbps = self.sample_bandwidth(t)
        bw_bps = max(bw_mbps * 1e6, 1.0)
        rtt = self.trace.rtt_at_time(self.trace_offset_s + t * self.inference_duration_s)
        up_ms = (uplink_kb * 8 * 1024 / bw_bps) * 1000 + rtt / 2
        down_ms = (downlink_kb * 8 * 1024 / bw_bps) * 1000 + rtt / 2
        return {'uplink_ms': up_ms, 'downlink_ms': down_ms,
                'total_ms': up_ms + down_ms, 'bandwidth_mbps': bw_mbps}


# ============================================================
# 3GPP-based synthetic trace generation (TR 38.901 inspired)
# ============================================================

class MobilityScenario(Enum):
    """3GPP mobility scenarios."""
    STATIC = "static"                # stationary user
    PEDESTRIAN = "pedestrian"        # 3 km/h, frequent handovers
    VEHICULAR = "vehicular"          # 60 km/h, fast fading
    HIGH_SPEED_TRAIN = "hst"         # 300 km/h, extreme Doppler


# Parameters: (mean_bw, std_bw, rtt_base, handover_rate_per_s, fade_depth_db)
_3GPP_SCENARIOS = {
    MobilityScenario.STATIC: {
        '4G': (15.0, 3.0, 40.0, 0.0, 3.0),
        '5G': (120.0, 25.0, 4.0, 0.0, 2.0),
        'WiFi': (40.0, 15.0, 8.0, 0.0, 5.0),
    },
    MobilityScenario.PEDESTRIAN: {
        '4G': (12.0, 5.0, 45.0, 0.01, 6.0),
        '5G': (100.0, 35.0, 5.0, 0.02, 4.0),
        'WiFi': (30.0, 18.0, 12.0, 0.05, 10.0),
    },
    MobilityScenario.VEHICULAR: {
        '4G': (8.0, 6.0, 55.0, 0.05, 10.0),
        '5G': (70.0, 40.0, 8.0, 0.08, 8.0),
        'WiFi': (15.0, 12.0, 20.0, 0.2, 15.0),
    },
    MobilityScenario.HIGH_SPEED_TRAIN: {
        '4G': (3.0, 4.0, 80.0, 0.15, 20.0),
        '5G': (40.0, 35.0, 15.0, 0.2, 15.0),
        'WiFi': (5.0, 5.0, 50.0, 0.5, 25.0),
    },
}


def generate_3gpp_trace(
    network_type: str = '4G',
    mobility: MobilityScenario = MobilityScenario.PEDESTRIAN,
    duration_s: float = 60.0,
    sample_interval_s: float = 0.1,
    seed: int = 42,
) -> NetworkTrace:
    """
    Generate a synthetic trace inspired by 3GPP TR 38.901 channel model.

    Models:
    - Rayleigh fading (fast component, correlated in time)
    - Shadow fading (slow component, log-normal)
    - Handover events (bandwidth drops)
    - RTT variation with load

    Args:
        network_type: '4G', '5G', or 'WiFi'
        mobility: MobilityScenario
        duration_s: trace duration in seconds
        sample_interval_s: measurement interval
        seed: random seed
    Returns:
        NetworkTrace
    """
    rng = np.random.RandomState(seed)
    params = _3GPP_SCENARIOS[mobility][network_type]
    mean_bw, std_bw, rtt_base, ho_rate, fade_depth_db = params

    num_samples = int(duration_s / sample_interval_s)
    times = np.linspace(0, duration_s, num_samples)

    # --- Shadow fading (slow, log-normal, AR(1)) ---
    shadow_ar = 0.995  # very slow
    shadow_std_db = fade_depth_db * 0.5
    shadow_db = np.zeros(num_samples)
    for i in range(1, num_samples):
        shadow_db[i] = (shadow_ar * shadow_db[i-1]
                        + np.sqrt(1 - shadow_ar**2) * rng.randn() * shadow_std_db)

    # --- Fast fading (Rayleigh envelope, correlated) ---
    # Doppler frequency determines correlation
    speed_map = {
        MobilityScenario.STATIC: 0,
        MobilityScenario.PEDESTRIAN: 3 / 3.6,      # m/s
        MobilityScenario.VEHICULAR: 60 / 3.6,
        MobilityScenario.HIGH_SPEED_TRAIN: 300 / 3.6,
    }
    v = speed_map[mobility]
    fc = 2e9 if network_type in ['4G', 'WiFi'] else 3.5e9
    c = 3e8
    f_doppler = max(v * fc / c, 0.1)

    # Generate correlated Rayleigh via filtered Gaussian
    ar_fast = np.exp(-2 * np.pi * f_doppler * sample_interval_s)
    fast_re = np.zeros(num_samples)
    fast_im = np.zeros(num_samples)
    for i in range(1, num_samples):
        fast_re[i] = ar_fast * fast_re[i-1] + np.sqrt(1 - ar_fast**2) * rng.randn()
        fast_im[i] = ar_fast * fast_im[i-1] + np.sqrt(1 - ar_fast**2) * rng.randn()
    fast_envelope_db = 10 * np.log10(fast_re**2 + fast_im**2 + 1e-6)
    # Normalize to zero mean
    fast_envelope_db -= np.mean(fast_envelope_db)
    fast_envelope_db *= (fade_depth_db / max(np.std(fast_envelope_db), 1e-3)) * 0.3

    # --- Handover events ---
    handover_mask = np.ones(num_samples)
    if ho_rate > 0:
        ho_interval = 1.0 / ho_rate  # average seconds between handovers
        next_ho = rng.exponential(ho_interval)
        for i in range(num_samples):
            if times[i] >= next_ho:
                # Handover: brief throughput drop for 200-500ms
                drop_duration = int(rng.uniform(0.2, 0.5) / sample_interval_s)
                end_idx = min(i + drop_duration, num_samples)
                handover_mask[i:end_idx] = rng.uniform(0.05, 0.3)
                next_ho = times[i] + rng.exponential(ho_interval)

    # --- Combine all effects ---
    total_fade_db = shadow_db + fast_envelope_db
    bw_linear = mean_bw * (10 ** (total_fade_db / 10)) * handover_mask

    # Clip to reasonable range
    bw_min = max(mean_bw * 0.02, 0.1)
    bw_max = mean_bw * 5.0
    bw_linear = np.clip(bw_linear, bw_min, bw_max)

    # --- RTT variation (inversely correlated with bandwidth, + jitter) ---
    bw_normalized = bw_linear / mean_bw
    rtt_values = rtt_base / np.sqrt(np.maximum(bw_normalized, 0.1))
    rtt_jitter = rng.randn(num_samples) * (rtt_base * 0.1)
    rtt_values = np.maximum(rtt_values + rtt_jitter, 1.0)

    # --- Build trace ---
    points = []
    for i in range(num_samples):
        points.append(TracePoint(
            time_s=times[i],
            bw_mbps=float(bw_linear[i]),
            rtt_ms=float(rtt_values[i]),
            loss_rate=0.0 if handover_mask[i] > 0.5 else 0.05,
        ))

    bws = [p.bw_mbps for p in points]
    return NetworkTrace(
        name=f"3GPP-{network_type}-{mobility.value}",
        points=points,
        duration_s=duration_s,
        mean_bw_mbps=float(np.mean(bws)),
        std_bw_mbps=float(np.std(bws)),
        source="3gpp",
    )


# ============================================================
# External trace file loading
# ============================================================

def load_trace_csv(filepath: str, time_col: str = 'time',
                   bw_col: str = 'throughput_mbps',
                   rtt_col: str = 'rtt_ms',
                   name: str = None) -> NetworkTrace:
    """
    Load a network trace from CSV file.

    Expected format: columns for time, throughput, and optionally RTT.
    Compatible with FCC MBA, Oboe, and custom measurement datasets.

    Args:
        filepath: path to CSV file
        time_col: column name for timestamps (seconds)
        bw_col: column name for throughput (Mbps)
        rtt_col: column name for RTT (ms), optional
        name: trace name (defaults to filename)
    Returns:
        NetworkTrace
    """
    points = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row[time_col])
            bw = float(row[bw_col])
            rtt = float(row.get(rtt_col, 50.0)) if rtt_col in row else 50.0
            loss = float(row.get('loss_rate', 0.0))
            points.append(TracePoint(t, bw, rtt, loss))

    points.sort(key=lambda p: p.time_s)
    bws = [p.bw_mbps for p in points]

    return NetworkTrace(
        name=name or os.path.basename(filepath),
        points=points,
        duration_s=points[-1].time_s - points[0].time_s if points else 0,
        mean_bw_mbps=float(np.mean(bws)) if bws else 0,
        std_bw_mbps=float(np.std(bws)) if bws else 0,
        source="custom",
    )


def load_trace_json(filepath: str, name: str = None) -> NetworkTrace:
    """
    Load a network trace from JSON file.

    Expected format:
    {
        "name": "trace_name",
        "points": [{"time_s": ..., "bw_mbps": ..., "rtt_ms": ...}, ...]
    }
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    points = []
    for p in data.get('points', []):
        points.append(TracePoint(
            time_s=p['time_s'],
            bw_mbps=p['bw_mbps'],
            rtt_ms=p.get('rtt_ms', 50.0),
            loss_rate=p.get('loss_rate', 0.0),
        ))
    points.sort(key=lambda p: p.time_s)
    bws = [p.bw_mbps for p in points]

    return NetworkTrace(
        name=name or data.get('name', os.path.basename(filepath)),
        points=points,
        duration_s=points[-1].time_s - points[0].time_s if points else 0,
        mean_bw_mbps=float(np.mean(bws)) if bws else 0,
        std_bw_mbps=float(np.std(bws)) if bws else 0,
        source=data.get('source', 'custom'),
    )


# ============================================================
# Markov-modulated trace (bandwidth regime switching)
# ============================================================

def generate_markov_trace(
    states: dict = None,
    transition_matrix: np.ndarray = None,
    duration_s: float = 60.0,
    sample_interval_s: float = 0.1,
    state_duration_s: float = 5.0,
    seed: int = 42,
) -> NetworkTrace:
    """
    Generate a Markov-modulated bandwidth trace.

    Models scenarios where the user moves between coverage areas
    (e.g., indoor → outdoor → subway → outdoor).

    Args:
        states: dict mapping state_name -> (mean_bw, std_bw, rtt)
        transition_matrix: K×K transition probability matrix
        duration_s: total trace duration
        sample_interval_s: sample interval
        state_duration_s: average duration in each state before transition check
        seed: random seed
    Returns:
        NetworkTrace
    """
    if states is None:
        states = {
            'outdoor_4G': (15.0, 5.0, 45.0),
            'indoor_WiFi': (35.0, 12.0, 10.0),
            'underground': (1.0, 0.8, 150.0),
            'congested_4G': (3.0, 2.0, 80.0),
        }
    state_names = list(states.keys())
    K = len(state_names)

    if transition_matrix is None:
        # Default: moderate switching
        transition_matrix = np.ones((K, K)) * 0.1 / (K - 1)
        np.fill_diagonal(transition_matrix, 0.9)
        # Normalize rows
        transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

    rng = np.random.RandomState(seed)
    num_samples = int(duration_s / sample_interval_s)
    times = np.linspace(0, duration_s, num_samples)

    # Generate state sequence
    current_state = 0
    state_seq = []
    check_interval = int(state_duration_s / sample_interval_s)

    for i in range(num_samples):
        if i > 0 and i % check_interval == 0:
            current_state = rng.choice(K, p=transition_matrix[current_state])
        state_seq.append(current_state)

    # Generate bandwidth within each state
    points = []
    for i in range(num_samples):
        s = state_seq[i]
        mean_bw, std_bw, rtt = states[state_names[s]]
        bw = max(0.1, rng.normal(mean_bw, std_bw))
        rtt_val = max(1.0, rng.normal(rtt, rtt * 0.15))
        points.append(TracePoint(times[i], bw, rtt_val, 0.0))

    bws = [p.bw_mbps for p in points]
    return NetworkTrace(
        name=f"Markov-{K}state",
        points=points,
        duration_s=duration_s,
        mean_bw_mbps=float(np.mean(bws)),
        std_bw_mbps=float(np.std(bws)),
        source="markov",
    )


# ============================================================
# Pre-built trace library for experiments
# ============================================================

def get_trace_library(seed: int = 42) -> dict:
    """
    Build a library of diverse network traces for comprehensive evaluation.

    Returns:
        dict mapping trace_name -> NetworkTrace
    """
    traces = {}

    # 3GPP traces across network types and mobility
    for net in ['4G', '5G', 'WiFi']:
        for mob in MobilityScenario:
            name = f"3gpp_{net}_{mob.value}"
            traces[name] = generate_3gpp_trace(
                net, mob, duration_s=60.0, seed=seed)
            seed += 1

    # Markov traces for mixed scenarios
    traces['markov_urban_commute'] = generate_markov_trace(
        states={
            'home_WiFi': (50.0, 15.0, 8.0),
            'outdoor_5G': (100.0, 30.0, 5.0),
            'subway_4G': (2.0, 1.5, 120.0),
            'office_WiFi': (60.0, 20.0, 6.0),
        },
        duration_s=60.0, state_duration_s=8.0, seed=seed,
    )
    seed += 1

    traces['markov_rural_drive'] = generate_markov_trace(
        states={
            'good_4G': (20.0, 8.0, 35.0),
            'weak_4G': (3.0, 2.0, 90.0),
            'no_signal': (0.2, 0.1, 500.0),
        },
        transition_matrix=np.array([
            [0.85, 0.12, 0.03],
            [0.10, 0.80, 0.10],
            [0.05, 0.15, 0.80],
        ]),
        duration_s=60.0, state_duration_s=5.0, seed=seed,
    )
    seed += 1

    # Bandwidth spike/drop scenarios (for AdaDC stress testing)
    traces['markov_bandwidth_spike'] = generate_markov_trace(
        states={
            'low_3G': (1.0, 0.5, 100.0),
            'high_WiFi': (50.0, 15.0, 8.0),
        },
        transition_matrix=np.array([
            [0.95, 0.05],
            [0.05, 0.95],
        ]),
        duration_s=60.0, state_duration_s=3.0, seed=seed,
    )

    return traces


# ============================================================
# Trace statistics and analysis
# ============================================================

def compute_trace_stats(trace: NetworkTrace) -> dict:
    """Compute comprehensive statistics for a network trace."""
    bws = np.array([p.bw_mbps for p in trace.points])
    rtts = np.array([p.rtt_ms for p in trace.points])

    # Coefficient of variation (measure of variability)
    cv = float(np.std(bws) / max(np.mean(bws), 1e-6))

    # Bandwidth changes (for scheduling analysis)
    if len(bws) > 1:
        bw_diffs = np.abs(np.diff(bws))
        change_rate = float(np.mean(bw_diffs) / max(np.mean(bws), 1e-6))
    else:
        change_rate = 0.0

    return {
        'name': trace.name,
        'duration_s': trace.duration_s,
        'num_points': trace.num_points,
        'bw_mean': float(np.mean(bws)),
        'bw_std': float(np.std(bws)),
        'bw_min': float(np.min(bws)),
        'bw_max': float(np.max(bws)),
        'bw_p5': float(np.percentile(bws, 5)),
        'bw_p50': float(np.percentile(bws, 50)),
        'bw_p95': float(np.percentile(bws, 95)),
        'bw_cv': cv,
        'bw_change_rate': change_rate,
        'rtt_mean': float(np.mean(rtts)),
        'rtt_p95': float(np.percentile(rtts, 95)),
    }
