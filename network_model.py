"""
Network simulation framework for edge-cloud collaborative inference.

Models realistic network conditions (3G/4G/WiFi/5G) with time-varying
bandwidth using log-normal distribution + AR(1) temporal correlation.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass


class NetworkProfile(Enum):
    """Predefined network profiles with typical parameters."""
    G3 = "3G"
    G4 = "4G"
    WIFI = "WiFi"
    G5 = "5G"


@dataclass
class NetworkParams:
    """Parameters for a network profile."""
    bw_mean_mbps: float      # mean bandwidth (Mbps)
    bw_std_mbps: float       # bandwidth std dev (Mbps)
    bw_min_mbps: float       # minimum bandwidth (Mbps)
    bw_max_mbps: float       # maximum bandwidth (Mbps)
    rtt_ms: float            # round-trip time (ms)
    jitter_ms: float         # jitter std dev (ms)
    ar_coeff: float          # AR(1) temporal correlation coefficient


# Realistic network profile parameters
NETWORK_PARAMS = {
    NetworkProfile.G3: NetworkParams(
        bw_mean_mbps=0.75, bw_std_mbps=0.2,
        bw_min_mbps=0.3, bw_max_mbps=1.5,
        rtt_ms=100.0, jitter_ms=20.0, ar_coeff=0.8,
    ),
    NetworkProfile.G4: NetworkParams(
        bw_mean_mbps=12.0, bw_std_mbps=5.0,
        bw_min_mbps=2.0, bw_max_mbps=30.0,
        rtt_ms=50.0, jitter_ms=10.0, ar_coeff=0.85,
    ),
    NetworkProfile.WIFI: NetworkParams(
        bw_mean_mbps=30.0, bw_std_mbps=12.0,
        bw_min_mbps=5.0, bw_max_mbps=80.0,
        rtt_ms=10.0, jitter_ms=3.0, ar_coeff=0.9,
    ),
    NetworkProfile.G5: NetworkParams(
        bw_mean_mbps=100.0, bw_std_mbps=40.0,
        bw_min_mbps=20.0, bw_max_mbps=300.0,
        rtt_ms=5.0, jitter_ms=1.0, ar_coeff=0.9,
    ),
}


class NetworkSimulator:
    """
    Simulates time-varying network bandwidth using AR(1) process
    in log-space for realistic bandwidth fluctuations.

    bandwidth(t) = exp(z(t)) where z(t) follows AR(1):
        z(t) = ar_coeff * z(t-1) + (1-ar_coeff) * mu + sigma * eps(t)

    The log-normal model ensures bandwidth stays positive and
    exhibits heavy-tailed fluctuations seen in real networks.
    """

    def __init__(self, profile: NetworkProfile, seed: int = 42):
        self.params = NETWORK_PARAMS[profile]
        self.profile = profile
        self.rng = np.random.RandomState(seed)

        # Convert to log-space parameters
        mean = self.params.bw_mean_mbps
        std = self.params.bw_std_mbps
        self.log_mu = np.log(mean**2 / np.sqrt(mean**2 + std**2))
        self.log_sigma = np.sqrt(np.log(1 + std**2 / mean**2))

        # AR(1) state
        self.z = self.log_mu
        self.ar = self.params.ar_coeff

        # Pre-generate bandwidth trace for reproducibility
        self._trace = None
        self._trace_len = 1000
        self._generate_trace()

    def _generate_trace(self):
        """Pre-generate bandwidth trace."""
        trace = np.zeros(self._trace_len)
        z = self.log_mu
        for i in range(self._trace_len):
            eps = self.rng.randn() * self.log_sigma * np.sqrt(1 - self.ar**2)
            z = self.ar * z + (1 - self.ar) * self.log_mu + eps
            bw = np.exp(z)
            bw = np.clip(bw, self.params.bw_min_mbps, self.params.bw_max_mbps)
            trace[i] = bw
        self._trace = trace

    def sample_bandwidth(self, t: float) -> float:
        """
        Get bandwidth at normalized time t ∈ [0, 1].
        Interpolates from pre-generated trace.

        Args:
            t: normalized time in [0, 1]
        Returns:
            bandwidth in Mbps
        """
        idx = min(int(t * (self._trace_len - 1)), self._trace_len - 1)
        return float(self._trace[idx])

    def get_bandwidth_trace(self, num_points: int = 100) -> tuple:
        """
        Get full bandwidth trace for visualization.

        Returns:
            (times, bandwidths) arrays
        """
        times = np.linspace(0, 1, num_points)
        bws = [self.sample_bandwidth(t) for t in times]
        return times, np.array(bws)

    def compute_transmission_time(self, data_size_kb: float,
                                   t: float = 0.5) -> float:
        """
        Compute transmission time for given data size at time t.

        transmission_time = data_bits / bandwidth + RTT/2 + jitter

        Args:
            data_size_kb: data size in KB
            t: normalized time for bandwidth lookup
        Returns:
            transmission time in milliseconds
        """
        bw_mbps = self.sample_bandwidth(t)
        data_bits = data_size_kb * 8 * 1024  # KB -> bits
        bw_bps = bw_mbps * 1e6               # Mbps -> bps

        transfer_ms = (data_bits / bw_bps) * 1000  # seconds -> ms
        rtt_half = self.params.rtt_ms / 2
        jitter = abs(self.rng.randn() * self.params.jitter_ms)

        return transfer_ms + rtt_half + jitter

    def compute_round_trip_transmission(self, uplink_kb: float,
                                         downlink_kb: float,
                                         t: float = 0.5) -> dict:
        """
        Compute full round-trip: edge→cloud (uplink) + cloud→edge (downlink).

        Args:
            uplink_kb: data sent from edge to cloud (KB)
            downlink_kb: data sent from cloud to edge (KB)
            t: normalized time
        Returns:
            dict with uplink_ms, downlink_ms, total_ms
        """
        bw_mbps = self.sample_bandwidth(t)
        bw_bps = bw_mbps * 1e6

        up_bits = uplink_kb * 8 * 1024
        down_bits = downlink_kb * 8 * 1024

        uplink_ms = (up_bits / bw_bps) * 1000 + self.params.rtt_ms / 2
        downlink_ms = (down_bits / bw_bps) * 1000 + self.params.rtt_ms / 2

        return {
            'uplink_ms': uplink_ms,
            'downlink_ms': downlink_ms,
            'total_ms': uplink_ms + downlink_ms,
            'bandwidth_mbps': bw_mbps,
        }


class BandwidthSwitchSimulator(NetworkSimulator):
    """
    Simulates bandwidth switching scenario: starts with one profile,
    switches to another at time t_switch.

    Useful for testing adaptive protocol under changing conditions.
    """

    def __init__(self, profile_before: NetworkProfile,
                 profile_after: NetworkProfile,
                 t_switch: float = 0.5,
                 seed: int = 42):
        self.sim_before = NetworkSimulator(profile_before, seed=seed)
        self.sim_after = NetworkSimulator(profile_after, seed=seed + 1)
        self.t_switch = t_switch
        self.profile = f"{profile_before.value}→{profile_after.value}"
        # Use before-profile params as default
        self.params = NETWORK_PARAMS[profile_before]

    def sample_bandwidth(self, t: float) -> float:
        if t < self.t_switch:
            return self.sim_before.sample_bandwidth(t / self.t_switch)
        else:
            t_rel = (t - self.t_switch) / (1 - self.t_switch)
            return self.sim_after.sample_bandwidth(t_rel)

    def compute_transmission_time(self, data_size_kb: float,
                                   t: float = 0.5) -> float:
        if t < self.t_switch:
            return self.sim_before.compute_transmission_time(data_size_kb,
                                                              t / self.t_switch)
        else:
            t_rel = (t - self.t_switch) / (1 - self.t_switch)
            return self.sim_after.compute_transmission_time(data_size_kb, t_rel)

    def compute_round_trip_transmission(self, uplink_kb: float,
                                         downlink_kb: float,
                                         t: float = 0.5) -> dict:
        if t < self.t_switch:
            return self.sim_before.compute_round_trip_transmission(
                uplink_kb, downlink_kb, t / self.t_switch)
        else:
            t_rel = (t - self.t_switch) / (1 - self.t_switch)
            return self.sim_after.compute_round_trip_transmission(
                uplink_kb, downlink_kb, t_rel)
