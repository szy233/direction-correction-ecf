"""
adadc_v2.py — Adaptive Direction Correction Protocol v2.

Core system contribution for INFOCOM:
  An online protocol that jointly optimizes generation quality, communication
  cost, and end-to-end latency under time-varying network conditions.

Key innovations over v1:
  1. Drift Estimator: monitors velocity divergence between edge/cloud to
     predict when correction is most beneficial
  2. Bandwidth Predictor: EWMA-based predictor for near-future bandwidth
  3. Joint Scheduler: decides (a) whether to query, (b) how many points,
     (c) compression config — all in real-time
  4. Hybrid Mode: seamlessly switches between DC and State Relay based on
     network conditions (DC when bandwidth is low, SR when high)

Protocol operates in three modes:
  - Offline: select strategy before generation (baseline)
  - Online-Greedy: decide at each candidate point using current bandwidth
  - Online-Optimal: look-ahead scheduling using predicted bandwidth

Compatible with both simulated and real network traces.
"""

import json
import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from compression import (compute_transmitted_size_kb, make_compress_fn,
                         compute_compression_ratio)
from latency_profiler import SystemLatencyProfile
from network_model import NetworkSimulator, NetworkProfile


# ============================================================
# Data structures
# ============================================================

class ProtocolMode(Enum):
    OFFLINE = "offline"
    ONLINE_GREEDY = "online_greedy"
    ONLINE_OPTIMAL = "online_optimal"


class InferenceMode(Enum):
    """What the edge is actually doing at each step."""
    EDGE_ONLY = "edge_only"
    DIRECTION_CORRECTION = "dc"
    STATE_RELAY = "state_relay"
    CLOUD_ONLY = "cloud_only"


@dataclass
class CorrectionConfig:
    """Configuration for a single correction event."""
    t_query: float                # when to query (normalized time)
    keep_ratio: float = 0.1      # sparsification ratio
    bits: int = 4                 # quantization bits
    mode: InferenceMode = InferenceMode.DIRECTION_CORRECTION

    @property
    def dv_size_kb(self) -> float:
        """Compressed delta_v size for default CIFAR-10 shape."""
        return compute_transmitted_size_kb((3, 32, 32), self.keep_ratio, self.bits)

    def dv_size_kb_for_shape(self, shape: tuple) -> float:
        """Compressed delta_v size for arbitrary shape."""
        return compute_transmitted_size_kb(shape, self.keep_ratio, self.bits)


@dataclass
class SchedulePlan:
    """Complete generation plan: a sequence of correction events."""
    corrections: list                       # list of CorrectionConfig
    fallback_mode: InferenceMode = InferenceMode.EDGE_ONLY
    total_steps: int = 20
    estimated_fid: float = 0.0
    estimated_latency_ms: float = 0.0
    estimated_comm_kb: float = 0.0
    label: str = ""

    @property
    def num_queries(self) -> int:
        return len(self.corrections)

    def summary(self) -> str:
        pts = [f"t={c.t_query:.2f}(kr={c.keep_ratio},b={c.bits})"
               for c in self.corrections]
        return f"Plan[{self.label}]: {len(pts)} queries at [{', '.join(pts)}]"


@dataclass
class DriftEstimate:
    """Result of drift estimation at a specific time."""
    t: float
    drift_magnitude: float      # ||v_edge_current - v_edge_ref|| / ||v_edge_ref||
    drift_direction_cos: float  # cosine similarity between current and reference
    should_correct: bool        # does drift exceed threshold?
    confidence: float           # confidence of the estimate [0, 1]


@dataclass
class ProtocolTrace:
    """Complete trace of protocol execution for analysis."""
    mode: str = "offline"
    schedule: Optional[SchedulePlan] = None
    events: list = field(default_factory=list)
    drift_history: list = field(default_factory=list)
    bandwidth_history: list = field(default_factory=list)

    # Cumulative metrics
    total_edge_compute_ms: float = 0.0
    total_cloud_compute_ms: float = 0.0
    total_uplink_ms: float = 0.0
    total_downlink_ms: float = 0.0
    total_comm_kb: float = 0.0
    actual_num_queries: int = 0
    fid: float = 0.0

    @property
    def total_latency_ms(self) -> float:
        return (self.total_edge_compute_ms + self.total_cloud_compute_ms
                + self.total_uplink_ms + self.total_downlink_ms)

    def to_dict(self) -> dict:
        return {
            'mode': self.mode,
            'schedule': self.schedule.summary() if self.schedule else None,
            'num_events': len(self.events),
            'total_edge_compute_ms': self.total_edge_compute_ms,
            'total_cloud_compute_ms': self.total_cloud_compute_ms,
            'total_uplink_ms': self.total_uplink_ms,
            'total_downlink_ms': self.total_downlink_ms,
            'total_latency_ms': self.total_latency_ms,
            'total_comm_kb': self.total_comm_kb,
            'actual_num_queries': self.actual_num_queries,
            'fid': self.fid,
            'drift_history': self.drift_history,
            'bandwidth_history': self.bandwidth_history,
        }


# ============================================================
# Drift Estimator
# ============================================================

class DriftEstimator:
    """
    Estimates velocity drift between edge and cloud models.

    Key insight: we can estimate drift WITHOUT querying the cloud by
    monitoring how fast the edge's OWN velocity is changing. If the edge
    velocity is changing rapidly, it likely diverges more from the cloud
    (which changes less due to higher capacity).

    Drift signal: d(t) = ||v_edge(x_t, t) - v_edge(x_{t-dt}, t-dt)|| / ||v_edge||

    This is a proxy for the true drift ||v_edge - v_cloud|| which we can't
    measure without communication.
    """

    def __init__(self, threshold: float = 0.3, window_size: int = 3):
        """
        Args:
            threshold: drift magnitude above which correction is recommended
            window_size: number of recent measurements for smoothing
        """
        self.threshold = threshold
        self.window_size = window_size
        self.history = []       # list of (t, v_edge_flat) tuples
        self.drift_values = []  # computed drift magnitudes

    def reset(self):
        self.history = []
        self.drift_values = []

    @torch.no_grad()
    def update(self, t: float, v_edge: torch.Tensor) -> DriftEstimate:
        """
        Update drift estimate with new velocity measurement.

        Args:
            t: current normalized time
            v_edge: edge velocity at current state [B, C, H, W]
        Returns:
            DriftEstimate with current drift assessment
        """
        v_flat = v_edge.detach().mean(dim=0).reshape(-1)  # average over batch
        v_norm = v_flat.norm().item()

        if len(self.history) == 0:
            self.history.append((t, v_flat.cpu()))
            return DriftEstimate(
                t=t, drift_magnitude=0.0, drift_direction_cos=1.0,
                should_correct=False, confidence=0.0)

        # Compute drift relative to first measurement (reference)
        v_ref = self.history[0][1]
        v_curr_cpu = v_flat.cpu()

        diff_norm = (v_curr_cpu - v_ref).norm().item()
        ref_norm = max(v_ref.norm().item(), 1e-8)
        drift_mag = diff_norm / ref_norm

        # Cosine similarity with reference
        cos_sim = float(torch.dot(v_curr_cpu, v_ref) / (v_curr_cpu.norm() * v_ref.norm() + 1e-8))

        self.drift_values.append(drift_mag)
        self.history.append((t, v_curr_cpu))

        # Smoothed drift (moving average)
        window = self.drift_values[-self.window_size:]
        smoothed = np.mean(window)

        # Confidence increases with more measurements
        confidence = min(len(self.drift_values) / 5.0, 1.0)

        should_correct = smoothed > self.threshold and confidence > 0.3

        return DriftEstimate(
            t=t,
            drift_magnitude=float(smoothed),
            drift_direction_cos=float(cos_sim),
            should_correct=should_correct,
            confidence=float(confidence),
        )

    def after_correction(self, t: float, v_corrected: torch.Tensor):
        """
        Reset reference after a correction is applied.
        The new corrected velocity becomes the reference for future drift.
        """
        v_flat = v_corrected.detach().mean(dim=0).reshape(-1).cpu()
        self.history = [(t, v_flat)]
        self.drift_values = []


# ============================================================
# Bandwidth Predictor
# ============================================================

class BandwidthPredictor:
    """
    EWMA-based bandwidth predictor for near-future estimation.

    Uses exponentially weighted moving average of recent bandwidth
    measurements. Provides both point prediction and uncertainty
    (conservative lower bound for latency estimation).
    """

    def __init__(self, alpha: float = 0.3, safety_factor: float = 0.7):
        """
        Args:
            alpha: EWMA smoothing factor (higher = more weight on recent)
            safety_factor: multiply prediction by this for conservative estimate
        """
        self.alpha = alpha
        self.safety_factor = safety_factor
        self.ewma = None
        self.ewma_var = None
        self.measurements = []

    def reset(self):
        self.ewma = None
        self.ewma_var = None
        self.measurements = []

    def update(self, bw_mbps: float):
        """Add a new bandwidth measurement."""
        self.measurements.append(bw_mbps)
        if self.ewma is None:
            self.ewma = bw_mbps
            self.ewma_var = 0.0
        else:
            self.ewma = self.alpha * bw_mbps + (1 - self.alpha) * self.ewma
            diff = bw_mbps - self.ewma
            self.ewma_var = self.alpha * diff**2 + (1 - self.alpha) * self.ewma_var

    def predict(self) -> float:
        """Predict next bandwidth (point estimate)."""
        return self.ewma if self.ewma is not None else 10.0

    def predict_conservative(self) -> float:
        """Conservative (lower bound) prediction for latency planning."""
        if self.ewma is None:
            return 5.0
        std = max(np.sqrt(self.ewma_var), 0.1)
        return max(self.ewma - std, 0.1) * self.safety_factor

    def predict_range(self) -> tuple:
        """Return (low, expected, high) bandwidth predictions."""
        if self.ewma is None:
            return (5.0, 10.0, 20.0)
        std = max(np.sqrt(self.ewma_var), 0.1)
        return (max(self.ewma - 2*std, 0.1), self.ewma, self.ewma + 2*std)


# ============================================================
# Latency Estimator
# ============================================================

class LatencyEstimator:
    """
    Estimates end-to-end latency for a given correction plan.

    Accounts for:
    - Edge compute (all steps run on edge)
    - Cloud compute (forward pass at query points)
    - Network transmission (uplink x_t* + downlink compressed δv)
    - Pipelining (cloud compute overlaps with edge compute for next segment)
    """

    def __init__(self, latency_profile: SystemLatencyProfile,
                 data_shape: tuple = (3, 32, 32)):
        self.profile = latency_profile
        self.data_shape = data_shape
        C, H, W = data_shape
        self.state_size_kb = C * H * W * 4 / 1024  # float32

    def estimate_plan_latency(self, plan: SchedulePlan,
                               bw_mbps: float, rtt_ms: float) -> dict:
        """
        Estimate total latency for a complete correction plan.

        For DC mode:
          - Edge runs ALL steps (continuous)
          - At each query point, communication happens IN PARALLEL with edge compute
          - But edge must WAIT for δv before applying correction

        Latency model:
          total = sum_segments(edge_compute_segment)
                  + sum_queries(max(0, comm_time - edge_overlap))
        """
        edge_total = plan.total_steps * self.profile.edge_step_ms
        cloud_total = 0.0
        uplink_total = 0.0
        downlink_total = 0.0
        comm_overhead = 0.0  # extra latency from communication stalls

        bw_bps = max(bw_mbps * 1e6, 1.0)

        for corr in plan.corrections:
            if corr.mode == InferenceMode.STATE_RELAY:
                # State relay: upload full state, cloud runs, download result
                up_ms = (self.state_size_kb * 8 * 1024 / bw_bps) * 1000 + rtt_ms / 2
                cloud_ms = self._cloud_steps_after(corr.t_query, plan.total_steps) \
                           * self.profile.cloud_step_ms
                down_ms = (self.state_size_kb * 8 * 1024 / bw_bps) * 1000 + rtt_ms / 2
                # In SR mode, edge STOPS after upload → no pipelining
                uplink_total += up_ms
                cloud_total += cloud_ms
                downlink_total += down_ms
                # Edge compute only up to t_query
                edge_total = int(plan.total_steps * corr.t_query) * self.profile.edge_step_ms

            elif corr.mode == InferenceMode.DIRECTION_CORRECTION:
                # DC: upload x_t* (or seed), cloud computes one forward pass, download δv
                dv_kb = corr.dv_size_kb_for_shape(self.data_shape)
                up_ms = (self.state_size_kb * 8 * 1024 / bw_bps) * 1000 + rtt_ms / 2
                cloud_ms = self.profile.cloud_query_ms
                down_ms = (dv_kb * 8 * 1024 / bw_bps) * 1000 + rtt_ms / 2

                # Communication can overlap with edge compute for the segment
                # after the query point. The edge only stalls if comm takes longer
                # than the edge steps in that overlap window.
                steps_after_query = max(1, int(plan.total_steps * (1 - corr.t_query)))
                edge_overlap_ms = min(3, steps_after_query) * self.profile.edge_step_ms
                comm_time = up_ms + cloud_ms + down_ms
                stall = max(0, comm_time - edge_overlap_ms)

                uplink_total += up_ms
                cloud_total += cloud_ms
                downlink_total += down_ms
                comm_overhead += stall

        total = edge_total + comm_overhead

        return {
            'edge_compute_ms': edge_total,
            'cloud_compute_ms': cloud_total,
            'uplink_ms': uplink_total,
            'downlink_ms': downlink_total,
            'comm_overhead_ms': comm_overhead,
            'total_ms': total,
            'bandwidth_mbps': bw_mbps,
        }

    def _cloud_steps_after(self, t_query: float, total_steps: int) -> int:
        return max(1, total_steps - int(total_steps * t_query))


# ============================================================
# Scheduling Strategies
# ============================================================

# Pre-computed FID lookup table (from experiments 1-5)
# Maps (num_queries, keep_ratio) -> expected FID delta vs edge-only
_FID_LOOKUP = {
    # (num_queries, keep_ratio, bits) -> FID
    (0, 0, 0): 63.0,          # edge-only
    (1, 0.05, 4): 62.2,
    (1, 0.10, 4): 60.0,
    (1, 0.20, 4): 57.6,
    (1, 0.50, 4): 54.6,
    (1, 1.00, 4): 53.9,
    (2, 0.05, 4): 57.7,
    (2, 0.10, 4): 54.5,
    (2, 0.20, 4): 50.7,
    (2, 0.50, 4): 46.3,
    (2, 1.00, 4): 45.3,
    (3, 0.05, 4): 56.7,
    (3, 0.10, 4): 52.8,
    (3, 0.20, 4): 48.2,
    (3, 0.50, 4): 42.8,
    (3, 1.00, 4): 41.6,
}

# State relay reference
_FID_STATE_RELAY = 36.8


def _lookup_fid(num_queries: int, keep_ratio: float, bits: int = 4) -> float:
    """Look up expected FID from pre-computed table."""
    key = (num_queries, keep_ratio, bits)
    if key in _FID_LOOKUP:
        return _FID_LOOKUP[key]
    # Interpolate: find nearest
    best_key = min(_FID_LOOKUP.keys(),
                   key=lambda k: abs(k[0] - num_queries) * 100
                                 + abs(k[1] - keep_ratio) * 10
                                 + abs(k[2] - bits))
    return _FID_LOOKUP[best_key]


class SchedulerStrategy:
    """
    Base class for scheduling strategies.

    A scheduler decides: given the current state (time, bandwidth,
    drift estimate, remaining budget), should we query the cloud?
    If yes, with what compression config?
    """

    # Candidate configurations to search over
    KEEP_RATIOS = [0.05, 0.1, 0.2, 0.5, 1.0]
    BITS = [4]
    QUERY_POINT_SETS = {
        1: [[0.7]],
        2: [[0.5, 0.8]],
        3: [[0.3, 0.6, 0.8]],
    }

    def plan(self, latency_budget_ms: float,
             bw_predictor: BandwidthPredictor,
             latency_est: LatencyEstimator,
             total_steps: int = 20,
             data_shape: tuple = (3, 32, 32)) -> SchedulePlan:
        """Generate a correction plan. Override in subclasses."""
        raise NotImplementedError


class OfflineScheduler(SchedulerStrategy):
    """
    Offline: exhaustively search all configs, pick best FID within budget.
    This is the v1 approach.
    """

    def plan(self, latency_budget_ms: float,
             bw_predictor: BandwidthPredictor,
             latency_est: LatencyEstimator,
             total_steps: int = 20,
             data_shape: tuple = (3, 32, 32)) -> SchedulePlan:

        bw = bw_predictor.predict_conservative()
        rtt = 50.0  # default RTT

        best_plan = SchedulePlan(
            corrections=[], total_steps=total_steps,
            estimated_fid=_lookup_fid(0, 0, 0),
            label="Edge Only")

        # Try all combinations
        for nq in [1, 2, 3]:
            for qpts in self.QUERY_POINT_SETS[nq]:
                for kr in self.KEEP_RATIOS:
                    corrections = [
                        CorrectionConfig(t_query=t, keep_ratio=kr, bits=4)
                        for t in qpts
                    ]
                    plan = SchedulePlan(
                        corrections=corrections, total_steps=total_steps,
                        estimated_fid=_lookup_fid(nq, kr, 4),
                        label=f"DC-{nq}pt-kr{kr}")
                    lat = latency_est.estimate_plan_latency(plan, bw, rtt)
                    plan.estimated_latency_ms = lat['total_ms']
                    plan.estimated_comm_kb = sum(
                        c.dv_size_kb_for_shape(data_shape) for c in corrections)

                    if lat['total_ms'] <= latency_budget_ms:
                        if plan.estimated_fid < best_plan.estimated_fid:
                            best_plan = plan

        return best_plan


class GreedyOnlineScheduler(SchedulerStrategy):
    """
    Online-Greedy: at each candidate point, decide based on current conditions.

    Decision logic:
    1. If drift is high AND bandwidth allows → query with max affordable compression
    2. If bandwidth is very high → consider state relay (better quality)
    3. Otherwise → skip (continue edge-only)
    """

    def __init__(self, drift_threshold: float = 0.3,
                 sr_bandwidth_threshold_mbps: float = 20.0):
        self.drift_threshold = drift_threshold
        self.sr_bw_threshold = sr_bandwidth_threshold_mbps

    def decide_at_point(self, t: float, drift: DriftEstimate,
                         bw_mbps: float, rtt_ms: float,
                         remaining_budget_ms: float,
                         latency_est: LatencyEstimator,
                         total_steps: int,
                         data_shape: tuple) -> Optional[CorrectionConfig]:
        """
        Real-time decision at a candidate query point.

        Returns CorrectionConfig if should query, None if should skip.
        """
        # Check if drift warrants correction
        if not drift.should_correct and drift.confidence > 0.5:
            return None

        # Check if we have enough latency budget for ANY query
        min_comm_time = self._min_query_latency(bw_mbps, rtt_ms, data_shape,
                                                  latency_est)
        remaining_edge_ms = max(1, int(total_steps * (1 - t))) * latency_est.profile.edge_step_ms

        if remaining_budget_ms < remaining_edge_ms + min_comm_time:
            return None  # no budget

        # Decide: DC or State Relay?
        if bw_mbps >= self.sr_bw_threshold:
            # High bandwidth → State Relay gives better quality
            sr_time = self._sr_latency(t, bw_mbps, rtt_ms, latency_est,
                                         total_steps, data_shape)
            if sr_time <= remaining_budget_ms:
                return CorrectionConfig(
                    t_query=t, keep_ratio=1.0, bits=32,
                    mode=InferenceMode.STATE_RELAY)

        # DC mode: find best compression that fits budget
        best_config = None
        best_fid = float('inf')

        for kr in self.KEEP_RATIOS:
            config = CorrectionConfig(t_query=t, keep_ratio=kr, bits=4)
            plan = SchedulePlan(corrections=[config], total_steps=total_steps)
            lat = latency_est.estimate_plan_latency(plan, bw_mbps, rtt_ms)

            if lat['total_ms'] <= remaining_budget_ms:
                fid = _lookup_fid(1, kr, 4)
                if fid < best_fid:
                    best_fid = fid
                    best_config = config

        return best_config

    def _min_query_latency(self, bw_mbps, rtt_ms, data_shape, lat_est):
        """Minimum latency for the cheapest possible query."""
        config = CorrectionConfig(t_query=0.5, keep_ratio=0.05, bits=4)
        plan = SchedulePlan(corrections=[config], total_steps=1)
        lat = lat_est.estimate_plan_latency(plan, bw_mbps, rtt_ms)
        return lat['comm_overhead_ms']

    def _sr_latency(self, t, bw_mbps, rtt_ms, lat_est, total_steps, data_shape):
        config = CorrectionConfig(t_query=t, mode=InferenceMode.STATE_RELAY)
        plan = SchedulePlan(corrections=[config], total_steps=total_steps)
        lat = lat_est.estimate_plan_latency(plan, bw_mbps, rtt_ms)
        return lat['total_ms']

    def plan(self, latency_budget_ms: float,
             bw_predictor: BandwidthPredictor,
             latency_est: LatencyEstimator,
             total_steps: int = 20,
             data_shape: tuple = (3, 32, 32)) -> SchedulePlan:
        """Pre-plan using predicted bandwidth (for comparison)."""
        bw = bw_predictor.predict()
        rtt = 50.0

        # Simulate decisions at candidate points with dummy drift
        corrections = []
        for t in [0.3, 0.5, 0.7, 0.8]:
            drift = DriftEstimate(t=t, drift_magnitude=0.5,
                                   drift_direction_cos=0.7,
                                   should_correct=True, confidence=0.8)
            remaining = latency_budget_ms - t * total_steps * latency_est.profile.edge_step_ms
            config = self.decide_at_point(
                t, drift, bw, rtt, remaining, latency_est, total_steps, data_shape)
            if config is not None:
                corrections.append(config)

        fid = _lookup_fid(len(corrections),
                          corrections[0].keep_ratio if corrections else 0, 4)

        return SchedulePlan(
            corrections=corrections, total_steps=total_steps,
            estimated_fid=fid, label=f"Greedy-{len(corrections)}pt")


class OptimalOnlineScheduler(SchedulerStrategy):
    """
    Online-Optimal: look-ahead scheduling using predicted bandwidth trajectory.

    Uses dynamic programming over the candidate query points to find
    the query schedule that minimizes expected FID within latency budget.

    State: (time_index, queries_made, budget_remaining)
    Action: query(kr) or skip
    Reward: -FID_improvement(queries_made + 1, kr)
    """

    def __init__(self, candidate_points: list = None):
        self.candidate_points = candidate_points or [0.3, 0.5, 0.7, 0.8]

    def plan(self, latency_budget_ms: float,
             bw_predictor: BandwidthPredictor,
             latency_est: LatencyEstimator,
             total_steps: int = 20,
             data_shape: tuple = (3, 32, 32)) -> SchedulePlan:
        """
        DP-based optimal scheduling.

        For each candidate point, predict bandwidth and enumerate
        (query with kr, skip) decisions. Find the sequence that
        minimizes FID while staying within latency budget.
        """
        N = len(self.candidate_points)
        bw_low, bw_mid, bw_high = bw_predictor.predict_range()
        rtt = 50.0

        # Use conservative bandwidth for planning
        bw = bw_predictor.predict_conservative()

        # DP: best FID achievable from point i onward with k queries already made
        # and remaining_budget_ms of latency budget
        best_global = None
        best_fid = _lookup_fid(0, 0, 0)  # edge-only baseline

        # Enumerate all subsets of candidate points (2^N, N<=4 so max 16)
        for mask in range(1 << N):
            selected = []
            for i in range(N):
                if mask & (1 << i):
                    selected.append(self.candidate_points[i])

            if not selected:
                continue

            nq = len(selected)

            # Try each compression level
            for kr in self.KEEP_RATIOS:
                corrections = [
                    CorrectionConfig(t_query=t, keep_ratio=kr, bits=4)
                    for t in selected
                ]
                plan = SchedulePlan(
                    corrections=corrections, total_steps=total_steps,
                    label=f"Opt-{nq}pt-kr{kr}")

                lat = latency_est.estimate_plan_latency(plan, bw, rtt)
                plan.estimated_latency_ms = lat['total_ms']
                plan.estimated_comm_kb = sum(
                    c.dv_size_kb_for_shape(data_shape) for c in corrections)

                if lat['total_ms'] > latency_budget_ms:
                    continue

                fid = _lookup_fid(nq, kr, 4)
                if fid < best_fid:
                    best_fid = fid
                    plan.estimated_fid = fid
                    best_global = plan

        if best_global is None:
            return SchedulePlan(
                corrections=[], total_steps=total_steps,
                estimated_fid=_lookup_fid(0, 0, 0),
                label="Edge Only (budget)")

        return best_global


class HybridScheduler(SchedulerStrategy):
    """
    Hybrid DC/SR Scheduler: the core INFOCOM contribution.

    Intelligently switches between Direction Correction and State Relay
    based on network conditions:

    - Low bandwidth (< threshold_low): DC with aggressive compression
      → minimize communication, accept some quality loss
    - Medium bandwidth: DC with moderate compression
      → balance quality and communication
    - High bandwidth (> threshold_high): State Relay
      → best quality when bandwidth permits
    - Very high bandwidth: Cloud-only offloading
      → let cloud handle everything

    This is the key insight: DC is NOT meant to replace SR in all cases.
    It's meant to extend the operating range to poor network conditions
    where SR is infeasible.
    """

    def __init__(self,
                 sr_threshold_mbps: float = 15.0,
                 cloud_threshold_mbps: float = 50.0,
                 dc_aggressive_threshold_mbps: float = 3.0):
        self.sr_threshold = sr_threshold_mbps
        self.cloud_threshold = cloud_threshold_mbps
        self.dc_aggressive_threshold = dc_aggressive_threshold_mbps

    def plan(self, latency_budget_ms: float,
             bw_predictor: BandwidthPredictor,
             latency_est: LatencyEstimator,
             total_steps: int = 20,
             data_shape: tuple = (3, 32, 32)) -> SchedulePlan:

        bw = bw_predictor.predict()
        bw_conservative = bw_predictor.predict_conservative()
        rtt = 50.0

        # --- Cloud-only if bandwidth is very high ---
        if bw_conservative >= self.cloud_threshold:
            return self._plan_cloud_only(latency_budget_ms, bw, rtt,
                                          latency_est, total_steps, data_shape)

        # --- State Relay if bandwidth is high ---
        if bw_conservative >= self.sr_threshold:
            plan = self._plan_state_relay(latency_budget_ms, bw, rtt,
                                           latency_est, total_steps, data_shape)
            if plan is not None:
                return plan

        # --- DC mode: select compression based on bandwidth ---
        if bw_conservative < self.dc_aggressive_threshold:
            # Very low bandwidth: 1 query, aggressive compression
            kr_candidates = [0.05, 0.1]
            nq_max = 1
        elif bw_conservative < self.sr_threshold:
            # Medium bandwidth: try multi-point with moderate compression
            kr_candidates = [0.1, 0.2, 0.5]
            nq_max = 3
        else:
            kr_candidates = self.KEEP_RATIOS
            nq_max = 3

        best_plan = SchedulePlan(
            corrections=[], total_steps=total_steps,
            estimated_fid=_lookup_fid(0, 0, 0),
            label="Edge Only")

        for nq in range(1, nq_max + 1):
            for qpts in self.QUERY_POINT_SETS.get(nq, []):
                for kr in kr_candidates:
                    corrections = [
                        CorrectionConfig(t_query=t, keep_ratio=kr, bits=4)
                        for t in qpts
                    ]
                    plan = SchedulePlan(
                        corrections=corrections, total_steps=total_steps,
                        label=f"Hybrid-DC-{nq}pt-kr{kr}")

                    lat = latency_est.estimate_plan_latency(plan, bw_conservative, rtt)
                    plan.estimated_latency_ms = lat['total_ms']
                    plan.estimated_comm_kb = sum(
                        c.dv_size_kb_for_shape(data_shape) for c in corrections)
                    plan.estimated_fid = _lookup_fid(nq, kr, 4)

                    if lat['total_ms'] <= latency_budget_ms:
                        if plan.estimated_fid < best_plan.estimated_fid:
                            best_plan = plan

        return best_plan

    def _plan_state_relay(self, budget, bw, rtt, lat_est, steps, shape):
        """Try state relay at optimal split point."""
        for t_star in [0.5, 0.4, 0.6, 0.3, 0.7]:
            corr = CorrectionConfig(t_query=t_star, mode=InferenceMode.STATE_RELAY)
            plan = SchedulePlan(
                corrections=[corr], total_steps=steps,
                estimated_fid=_FID_STATE_RELAY,
                label=f"Hybrid-SR-t{t_star}")
            lat = lat_est.estimate_plan_latency(plan, bw, rtt)
            plan.estimated_latency_ms = lat['total_ms']
            C, H, W = shape
            plan.estimated_comm_kb = 2 * C * H * W * 4 / 1024
            if lat['total_ms'] <= budget:
                return plan
        return None

    def _plan_cloud_only(self, budget, bw, rtt, lat_est, steps, shape):
        """Full cloud offloading."""
        C, H, W = shape
        state_kb = C * H * W * 4 / 1024
        bw_bps = max(bw * 1e6, 1.0)
        up = (state_kb * 8 * 1024 / bw_bps) * 1000 + rtt / 2
        cloud = steps * lat_est.profile.cloud_step_ms
        down = (state_kb * 8 * 1024 / bw_bps) * 1000 + rtt / 2
        total = up + cloud + down

        return SchedulePlan(
            corrections=[CorrectionConfig(t_query=0.0, mode=InferenceMode.CLOUD_ONLY)],
            total_steps=steps,
            estimated_fid=27.8,
            estimated_latency_ms=total,
            estimated_comm_kb=2 * state_kb,
            label="Hybrid-CloudOnly")


# ============================================================
# AdaDC v2 Protocol
# ============================================================

class AdaDCv2Protocol:
    """
    Adaptive Direction Correction Protocol v2.

    Integrates drift estimation, bandwidth prediction, latency estimation,
    and scheduling into a unified online protocol.

    Usage:
        protocol = AdaDCv2Protocol(latency_profile, network_sim)

        # Offline mode
        plan = protocol.plan_offline(budget_ms=2000)

        # Online mode (call during generation)
        protocol.start_generation(budget_ms=2000)
        for step in range(total_steps):
            t = step / total_steps
            v_edge = edge_model(x_t, t)
            decision = protocol.step(t, v_edge, x_t)
            if decision.should_query:
                # do the query...
                protocol.report_correction(t, v_corrected)
        trace = protocol.finish_generation()
    """

    def __init__(self,
                 latency_profile: SystemLatencyProfile,
                 network_sim,                    # NetworkSimulator or TraceReplaySimulator
                 data_shape: tuple = (3, 32, 32),
                 scheduler: str = 'hybrid',      # 'offline', 'greedy', 'optimal', 'hybrid'
                 drift_threshold: float = 0.3,
                 total_steps: int = 20):

        self.latency_profile = latency_profile
        self.network = network_sim
        self.data_shape = data_shape
        self.total_steps = total_steps

        self.drift_estimator = DriftEstimator(threshold=drift_threshold)
        self.bw_predictor = BandwidthPredictor()
        self.latency_est = LatencyEstimator(latency_profile, data_shape)

        # Select scheduler
        scheduler_map = {
            'offline': OfflineScheduler(),
            'greedy': GreedyOnlineScheduler(drift_threshold=drift_threshold),
            'optimal': OptimalOnlineScheduler(),
            'hybrid': HybridScheduler(),
        }
        self.scheduler = scheduler_map.get(scheduler, HybridScheduler())
        self.scheduler_name = scheduler

        # Runtime state
        self._trace = None
        self._budget_ms = 0
        self._elapsed_ms = 0
        self._plan = None

        # Candidate query points for online mode
        self._candidate_points = [0.3, 0.5, 0.7, 0.8]
        self._query_tolerance = 0.5 / total_steps  # how close t must be to a candidate

    def plan_offline(self, budget_ms: float) -> SchedulePlan:
        """Pre-plan before generation starts."""
        # Sample bandwidth trajectory for prediction
        for t in np.linspace(0, 1, 20):
            bw = self.network.sample_bandwidth(t)
            self.bw_predictor.update(bw)

        plan = self.scheduler.plan(
            budget_ms, self.bw_predictor, self.latency_est,
            self.total_steps, self.data_shape)
        return plan

    def start_generation(self, budget_ms: float):
        """Initialize for online generation."""
        self._budget_ms = budget_ms
        self._elapsed_ms = 0
        self._trace = ProtocolTrace(mode=self.scheduler_name)
        self.drift_estimator.reset()
        self.bw_predictor.reset()
        self._plan = None
        # Initial bandwidth samples
        bw_init = self.network.sample_bandwidth(0.0)
        self.bw_predictor.update(bw_init)

    @dataclass
    class StepDecision:
        should_query: bool = False
        config: Optional[CorrectionConfig] = None
        drift: Optional[DriftEstimate] = None
        bandwidth_mbps: float = 0.0

    def step(self, t: float, v_edge: torch.Tensor,
             x_t: torch.Tensor = None) -> 'AdaDCv2Protocol.StepDecision':
        """
        Called at each Euler step during generation.

        Returns decision: should we query the cloud at this point?
        """
        # Update bandwidth
        bw = self.network.sample_bandwidth(t)
        self.bw_predictor.update(bw)

        # Update drift
        drift = self.drift_estimator.update(t, v_edge)

        # Track edge compute time
        self._elapsed_ms += self.latency_profile.edge_step_ms

        # Record
        if self._trace:
            self._trace.drift_history.append({
                't': t, 'drift': drift.drift_magnitude,
                'cos_sim': drift.drift_direction_cos})
            self._trace.bandwidth_history.append({
                't': t, 'bw_mbps': bw})

        # Check if t is near a candidate query point
        is_candidate = any(abs(t - cp) < self._query_tolerance
                           for cp in self._candidate_points)

        if not is_candidate:
            return self.StepDecision(
                should_query=False, drift=drift, bandwidth_mbps=bw)

        # At candidate point: make decision
        remaining_budget = self._budget_ms - self._elapsed_ms

        if isinstance(self.scheduler, GreedyOnlineScheduler):
            rtt = getattr(self.network.params, 'rtt_ms', 50.0)
            config = self.scheduler.decide_at_point(
                t, drift, bw, rtt, remaining_budget,
                self.latency_est, self.total_steps, self.data_shape)
        else:
            # For other schedulers, use plan
            if self._plan is None:
                self._plan = self.scheduler.plan(
                    remaining_budget, self.bw_predictor, self.latency_est,
                    self.total_steps, self.data_shape)
                if self._trace:
                    self._trace.schedule = self._plan

            # Check if current t is in the plan
            config = None
            for corr in self._plan.corrections:
                if abs(t - corr.t_query) < self._query_tolerance:
                    config = corr
                    break

        if config is not None and self._trace:
            # Estimate and record communication cost
            rtt = getattr(self.network.params, 'rtt_ms', 50.0)
            plan_tmp = SchedulePlan(corrections=[config], total_steps=self.total_steps)
            lat = self.latency_est.estimate_plan_latency(plan_tmp, bw, rtt)
            self._elapsed_ms += lat['comm_overhead_ms']
            self._trace.total_comm_kb += config.dv_size_kb_for_shape(self.data_shape)
            self._trace.actual_num_queries += 1
            self._trace.events.append({
                't': t, 'action': 'query',
                'mode': config.mode.value,
                'keep_ratio': config.keep_ratio,
                'bw_mbps': bw, 'drift': drift.drift_magnitude,
                'comm_overhead_ms': lat['comm_overhead_ms'],
            })

        return self.StepDecision(
            should_query=config is not None,
            config=config, drift=drift, bandwidth_mbps=bw)

    def report_correction(self, t: float, v_corrected: torch.Tensor):
        """Report that correction was applied (resets drift estimator)."""
        self.drift_estimator.after_correction(t, v_corrected)

    def finish_generation(self) -> ProtocolTrace:
        """Finalize and return the complete protocol trace."""
        if self._trace:
            self._trace.total_edge_compute_ms = (
                self.total_steps * self.latency_profile.edge_step_ms)
        return self._trace

    # ---- Batch simulation (no actual model inference) ----

    def simulate_batch(self, num_runs: int = 100,
                       budget_ms: float = 2000.0,
                       trace_offsets: list = None) -> list:
        """
        Simulate protocol decisions for many runs without actual inference.
        Useful for statistical analysis of protocol behavior.

        Args:
            num_runs: number of simulation runs
            budget_ms: latency budget per run
            trace_offsets: if using TraceReplaySimulator, vary start times
        Returns:
            list of ProtocolTrace (one per run)
        """
        traces = []
        for i in range(num_runs):
            # Vary network conditions
            if trace_offsets and hasattr(self.network, 'trace_offset_s'):
                self.network.trace_offset_s = trace_offsets[i % len(trace_offsets)]

            plan = self.plan_offline(budget_ms)
            trace = ProtocolTrace(
                mode=self.scheduler_name,
                schedule=plan,
                total_edge_compute_ms=self.total_steps * self.latency_profile.edge_step_ms,
                total_comm_kb=plan.estimated_comm_kb,
                actual_num_queries=plan.num_queries,
            )

            # Estimate latency using predicted bandwidth
            bw = self.bw_predictor.predict_conservative()
            rtt = getattr(self.network.params, 'rtt_ms', 50.0)
            lat = self.latency_est.estimate_plan_latency(plan, bw, rtt)
            trace.total_cloud_compute_ms = lat['cloud_compute_ms']
            trace.total_uplink_ms = lat['uplink_ms']
            trace.total_downlink_ms = lat['downlink_ms']

            # Record bandwidth trajectory
            for t in np.linspace(0, 1, self.total_steps):
                bw_t = self.network.sample_bandwidth(t)
                trace.bandwidth_history.append({'t': float(t), 'bw_mbps': bw_t})

            traces.append(trace)

        return traces


# ============================================================
# Convenience: build protocol from components
# ============================================================

def build_adadc_v2(
    latency_profile: SystemLatencyProfile = None,
    network_sim=None,
    data_shape: tuple = (3, 32, 32),
    scheduler: str = 'hybrid',
    total_steps: int = 20,
) -> AdaDCv2Protocol:
    """
    Build a complete AdaDC v2 protocol instance.

    Args:
        latency_profile: measured or simulated system latency
        network_sim: NetworkSimulator or TraceReplaySimulator
        data_shape: tensor shape (C, H, W) for communication cost
        scheduler: 'offline', 'greedy', 'optimal', 'hybrid'
        total_steps: total Euler steps
    Returns:
        Configured AdaDCv2Protocol
    """
    if latency_profile is None:
        from latency_profiler import SIMULATED_PROFILES
        latency_profile = SIMULATED_PROFILES['mobile_small']

    if network_sim is None:
        network_sim = NetworkSimulator(NetworkProfile.G4, seed=42)

    return AdaDCv2Protocol(
        latency_profile=latency_profile,
        network_sim=network_sim,
        data_shape=data_shape,
        scheduler=scheduler,
        total_steps=total_steps,
    )
