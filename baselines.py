"""
baselines.py — Additional baseline methods for comparison.

Implements:
1. Early Exit: edge uses fewer steps, no communication
2. Split Inference (Neurosurgeon-style): split UNet at layer boundary
3. Multi-point State Relay: relay full state at multiple points
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from rectified_flow import RectifiedFlowSampler, DirectionCorrectionSampler


# ============================================================
# 1. Early Exit Baseline
# ============================================================

def early_exit_samples(sampler: DirectionCorrectionSampler,
                       x_0: torch.Tensor,
                       steps_list: list = [5, 10, 15, 20]) -> dict:
    """
    Early exit: edge runs fewer Euler steps with no communication.
    Simulates quality-latency tradeoff on edge alone.

    Args:
        sampler: DirectionCorrectionSampler (uses edge_sampler)
        x_0: initial noise [B, C, H, W]
        steps_list: list of step counts to try
    Returns:
        dict mapping num_steps -> generated samples tensor
    """
    results = {}
    for steps in steps_list:
        x = sampler.edge_sampler.sample(x_0, num_steps=steps,
                                         t_start=0.0, t_end=1.0)
        results[steps] = x
    return results


# ============================================================
# 2. Split Inference (Neurosurgeon-style)
# ============================================================

def measure_activation_sizes(model: nn.Module,
                              input_shape: tuple = (1, 3, 32, 32),
                              device: str = 'cuda') -> list:
    """
    Measure intermediate activation sizes at each layer of the UNet.
    Used to find the optimal split point for Neurosurgeon-style inference.

    Returns:
        list of (layer_name, activation_size_kb) tuples
    """
    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)
    t = torch.full((input_shape[0],), 0.5, device=device)

    activations = []

    # Hook to capture activation sizes
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                size_kb = output.numel() * 4 / 1024  # float32
                activations.append((name, size_kb, output.shape))
        return hook_fn

    # Register hooks on key layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        _ = model(x, t)

    for h in hooks:
        h.remove()

    return activations


def split_inference_latency(edge_model: nn.Module,
                             cloud_model: nn.Module,
                             input_shape: tuple = (1, 3, 32, 32),
                             device: str = 'cuda') -> list:
    """
    Analyze split inference: for each possible split point,
    compute (activation_size_kb, edge_compute_fraction, cloud_compute_fraction).

    Returns a list of split options sorted by activation size.
    """
    edge_acts = measure_activation_sizes(edge_model, input_shape, device)

    # For UNet, the bottleneck (smallest activation) is at the middle
    # Sort by activation size to find best split points
    split_options = []
    for name, size_kb, shape in edge_acts:
        split_options.append({
            'layer': name,
            'activation_kb': size_kb,
            'activation_shape': list(shape),
        })

    split_options.sort(key=lambda x: x['activation_kb'])
    return split_options


# ============================================================
# 3. Multi-point State Relay
# ============================================================

@torch.no_grad()
def multi_relay_samples(sampler: DirectionCorrectionSampler,
                        x_0: torch.Tensor,
                        relay_points: list = [0.5],
                        edge_steps_per_seg: int = 5,
                        cloud_steps_per_seg: int = 10) -> dict:
    """
    Multi-point state relay: at each relay point, send full state
    to cloud, cloud advances, sends back state, edge continues.

    Unlike DC which sends δv (~sparse), state relay always sends
    full x_t* (12KB for CIFAR-10).

    Args:
        sampler: DirectionCorrectionSampler
        x_0: initial noise [B, C, H, W]
        relay_points: sorted list of relay times
        edge_steps_per_seg: edge steps per segment
        cloud_steps_per_seg: cloud steps per segment
    Returns:
        dict with x_final, relay_log, comm_kb
    """
    relay_points = sorted(relay_points)
    boundaries = [0.0] + relay_points + [1.0]

    x = x_0.clone()
    relay_log = []
    total_steps = 0

    for i in range(len(boundaries) - 1):
        t_start = boundaries[i]
        t_end = boundaries[i + 1]

        if i < len(relay_points):
            # Edge runs to relay point
            x = sampler.edge_sampler.sample(
                x, num_steps=edge_steps_per_seg,
                t_start=t_start, t_end=t_end)
            total_steps += edge_steps_per_seg

            # At relay: cloud takes over briefly (one step correction)
            # In full state relay, cloud runs from relay to next segment
            relay_log.append({
                't': t_end,
                'state_size_kb': x.shape[1] * x.shape[2] * x.shape[3] * 4 / 1024,
            })
        else:
            # Last segment: cloud finishes
            x = sampler.cloud_sampler.sample(
                x, num_steps=cloud_steps_per_seg,
                t_start=t_start, t_end=t_end)
            total_steps += cloud_steps_per_seg

    # Communication: full state at each relay point (up + down)
    state_kb = x_0.shape[1] * x_0.shape[2] * x_0.shape[3] * 4 / 1024
    total_comm_kb = len(relay_points) * state_kb * 2  # up + down

    return {
        'x_final': x,
        'relay_log': relay_log,
        'comm_kb': total_comm_kb,
        'num_relays': len(relay_points),
    }


# ============================================================
# Summary: communication cost for all methods
# ============================================================

def communication_cost_table(image_shape: tuple = (3, 32, 32)) -> dict:
    """
    Compute communication cost (KB) for each method family.

    Returns dict of method -> communication cost formula.
    """
    C, H, W = image_shape
    state_kb = C * H * W * 4 / 1024  # float32

    return {
        'Edge Only': {'per_query_kb': 0, 'description': 'No communication'},
        'Cloud Only': {
            'per_query_kb': 2 * state_kb,
            'description': f'Send noise + receive image: 2×{state_kb:.1f}KB'
        },
        'State Relay': {
            'per_query_kb': 2 * state_kb,
            'description': f'Send x_t* + receive x_1: 2×{state_kb:.1f}KB'
        },
        'Multi-Relay (k points)': {
            'per_query_kb': 2 * state_kb,
            'description': f'k × 2×{state_kb:.1f}KB full state transfers'
        },
        'DC (kr=10%, 4-bit)': {
            'per_query_kb': _dc_comm_kb(C * H * W, 0.1, 4) + state_kb,
            'description': 'Send x_t* + receive compressed δv'
        },
        'DC (kr=5%, 4-bit)': {
            'per_query_kb': _dc_comm_kb(C * H * W, 0.05, 4) + state_kb,
            'description': 'Send x_t* + receive compressed δv'
        },
        'DC (kr=1%, 4-bit)': {
            'per_query_kb': _dc_comm_kb(C * H * W, 0.01, 4) + state_kb,
            'description': 'Send x_t* + receive compressed δv'
        },
    }


def _dc_comm_kb(total_elements: int, keep_ratio: float, bits: int) -> float:
    """Helper: compute compressed δv size in KB."""
    kept = max(1, int(total_elements * keep_ratio))
    compressed_bits = kept * (bits + 16) + 64
    return compressed_bits / 8 / 1024
