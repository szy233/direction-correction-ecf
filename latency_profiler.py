"""
Computation latency profiler for edge and cloud models.

Measures per-step inference time using CUDA events for accurate GPU timing.
Supports simulated latency scaling for analyzing larger model scenarios.
"""

import time
import torch
import numpy as np
from dataclasses import dataclass, field


@dataclass
class LatencyStats:
    """Statistics from latency profiling."""
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    num_runs: int
    raw_ms: list = field(default_factory=list, repr=False)

    @classmethod
    def from_measurements(cls, times_ms: list) -> 'LatencyStats':
        arr = np.array(times_ms)
        return cls(
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            num_runs=len(times_ms),
            raw_ms=times_ms,
        )


class LatencyProfiler:
    """Profile model inference latency on GPU or CPU."""

    @staticmethod
    def profile_model(model: torch.nn.Module,
                      input_shape: tuple = (1, 3, 32, 32),
                      device: str = 'cuda',
                      num_warmup: int = 10,
                      num_runs: int = 50) -> LatencyStats:
        """
        Measure single forward pass latency.

        Args:
            model: the model to profile
            input_shape: (B, C, H, W) input tensor shape
            device: 'cuda' or 'cpu'
            num_warmup: warmup iterations
            num_runs: measurement iterations
        Returns:
            LatencyStats with timing statistics
        """
        model = model.to(device).eval()
        x = torch.randn(*input_shape, device=device)
        t = torch.full((input_shape[0],), 0.5, device=device)

        use_cuda = device == 'cuda' and torch.cuda.is_available()

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(x, t)
                if use_cuda:
                    torch.cuda.synchronize()

        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if use_cuda:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    _ = model(x, t)
                    end_event.record()
                    torch.cuda.synchronize()
                    times.append(start_event.elapsed_time(end_event))
                else:
                    start = time.perf_counter()
                    _ = model(x, t)
                    elapsed = (time.perf_counter() - start) * 1000
                    times.append(elapsed)

        return LatencyStats.from_measurements(times)


@dataclass
class SystemLatencyProfile:
    """Complete latency profile for edge-cloud system."""
    edge_step_ms: float       # one Euler step on edge model
    cloud_step_ms: float      # one Euler step on cloud model
    cloud_query_ms: float     # one forward pass on cloud (for delta_v)
    edge_params: int          # edge model parameter count
    cloud_params: int         # cloud model parameter count

    def scale_to_params(self, target_edge_params: int,
                        target_cloud_params: int) -> 'SystemLatencyProfile':
        """
        Estimate latency for larger models by linear scaling.
        Useful for scalability analysis without retraining.
        """
        edge_scale = target_edge_params / self.edge_params
        cloud_scale = target_cloud_params / self.cloud_params
        return SystemLatencyProfile(
            edge_step_ms=self.edge_step_ms * edge_scale,
            cloud_step_ms=self.cloud_step_ms * cloud_scale,
            cloud_query_ms=self.cloud_query_ms * cloud_scale,
            edge_params=target_edge_params,
            cloud_params=target_cloud_params,
        )

    def to_dict(self) -> dict:
        return {
            'edge_step_ms': self.edge_step_ms,
            'cloud_step_ms': self.cloud_step_ms,
            'cloud_query_ms': self.cloud_query_ms,
            'edge_params': self.edge_params,
            'cloud_params': self.cloud_params,
        }


def profile_system(edge_model: torch.nn.Module,
                   cloud_model: torch.nn.Module,
                   input_shape: tuple = (1, 3, 32, 32),
                   device: str = 'cuda',
                   num_runs: int = 50) -> SystemLatencyProfile:
    """
    Profile complete edge-cloud system.

    Args:
        edge_model: edge UNet model
        cloud_model: cloud UNet model
        input_shape: input tensor shape
        device: device to profile on
        num_runs: number of measurement iterations
    Returns:
        SystemLatencyProfile with all timing info
    """
    profiler = LatencyProfiler()

    edge_stats = profiler.profile_model(
        edge_model, input_shape, device, num_runs=num_runs)
    cloud_stats = profiler.profile_model(
        cloud_model, input_shape, device, num_runs=num_runs)

    edge_params = sum(p.numel() for p in edge_model.parameters())
    cloud_params = sum(p.numel() for p in cloud_model.parameters())

    return SystemLatencyProfile(
        edge_step_ms=edge_stats.mean_ms,
        cloud_step_ms=cloud_stats.mean_ms,
        cloud_query_ms=cloud_stats.mean_ms,  # same as one forward pass
        edge_params=edge_params,
        cloud_params=cloud_params,
    )


# Predefined simulated profiles for large-scale analysis
SIMULATED_PROFILES = {
    'mobile_small': SystemLatencyProfile(
        edge_step_ms=50.0, cloud_step_ms=5.0, cloud_query_ms=5.0,
        edge_params=4_000_000, cloud_params=35_000_000,
    ),
    'mobile_medium': SystemLatencyProfile(
        edge_step_ms=150.0, cloud_step_ms=10.0, cloud_query_ms=10.0,
        edge_params=50_000_000, cloud_params=500_000_000,
    ),
    'mobile_large': SystemLatencyProfile(
        edge_step_ms=500.0, cloud_step_ms=20.0, cloud_query_ms=20.0,
        edge_params=200_000_000, cloud_params=2_000_000_000,
    ),
    'sd_on_phone': SystemLatencyProfile(
        edge_step_ms=800.0, cloud_step_ms=30.0, cloud_query_ms=30.0,
        edge_params=500_000_000, cloud_params=3_000_000_000,
    ),
}
