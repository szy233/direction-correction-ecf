"""
compression.py — Compression strategies for the direction correction vector δv.

Two-layer compression:
  Layer 1: Threshold sparsification (keep top-k% of entries by magnitude)
  Layer 2: Scalar quantization (reduce to N-bit representation)

These two layers together can compress δv to 1%–5% of the original x_{t*} size.
"""

import torch
import numpy as np
from typing import Optional


def sparsify(delta_v: torch.Tensor, keep_ratio: float = 0.1) -> torch.Tensor:
    """
    Threshold sparsification: keep only the top-k% entries by absolute value.

    Args:
        delta_v:    input tensor [B, C, H, W]
        keep_ratio: fraction of entries to keep (e.g., 0.1 = top 10%)
    Returns:
        sparse_v:   tensor with same shape, small entries zeroed out
    """
    B = delta_v.shape[0]
    flat = delta_v.reshape(B, -1)
    k = max(1, int(flat.shape[1] * keep_ratio))

    # Find the k-th largest absolute value per sample
    topk_vals, _ = torch.topk(flat.abs(), k, dim=1)
    threshold = topk_vals[:, -1:]  # [B, 1]

    # Zero out entries below threshold
    mask = flat.abs() >= threshold
    sparse_flat = flat * mask.float()
    return sparse_flat.reshape_as(delta_v)


def quantize(delta_v: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """
    Uniform scalar quantization to N-bit.

    Maps values to [0, 2^bits - 1] uniformly within [min, max] per sample,
    then dequantizes back. Simulates the information loss of low-bit transmission.

    Args:
        delta_v: input tensor [B, C, H, W]
        bits:    number of quantization bits (4, 8, etc.)
    Returns:
        dequantized tensor (same shape, with quantization noise)
    """
    B = delta_v.shape[0]
    flat = delta_v.reshape(B, -1)
    levels = 2 ** bits - 1

    # Per-sample min/max
    vmin = flat.min(dim=1, keepdim=True).values
    vmax = flat.max(dim=1, keepdim=True).values
    scale = (vmax - vmin).clamp(min=1e-8)

    # Quantize
    normalized = (flat - vmin) / scale  # [0, 1]
    quantized = torch.round(normalized * levels) / levels
    dequantized = quantized * scale + vmin

    return dequantized.reshape_as(delta_v)


def compress_delta_v(
    delta_v: torch.Tensor,
    keep_ratio: float = 0.1,
    bits: int = 8,
) -> torch.Tensor:
    """
    Two-layer compression: sparsify then quantize.

    Args:
        delta_v:    direction correction vector [B, C, H, W]
        keep_ratio: sparsification ratio (e.g., 0.1 = keep top 10%)
        bits:       quantization bits
    Returns:
        compressed δv (same shape, with information loss)
    """
    sparse_v = sparsify(delta_v, keep_ratio)
    compressed_v = quantize(sparse_v, bits)
    return compressed_v


def compute_compression_ratio(
    delta_v: torch.Tensor,
    keep_ratio: float = 0.1,
    bits: int = 8,
) -> dict:
    """
    Compute the actual compression ratio vs transmitting the full x_{t*}.

    For sparse + quantized δv:
        - Only non-zero entries need to be sent
        - Each entry uses `bits` instead of 32
        - Plus indices (16-bit each for CIFAR-10 sizes)

    Returns:
        dict with compression statistics
    """
    B, C, H, W = delta_v.shape
    total_elements = C * H * W
    kept_elements = int(total_elements * keep_ratio)

    # Original x_{t*}: 32-bit per element
    original_bits = total_elements * 32

    # Compressed δv: value (N-bit) + index (16-bit) per kept element
    # Plus per-sample min/max (2 × 32-bit)
    compressed_bits = kept_elements * (bits + 16) + 2 * 32

    ratio = compressed_bits / original_bits

    return {
        'original_size_bits': original_bits,
        'compressed_size_bits': compressed_bits,
        'compression_ratio': ratio,
        'original_size_kb': original_bits / 8 / 1024,
        'compressed_size_kb': compressed_bits / 8 / 1024,
        'keep_ratio': keep_ratio,
        'quant_bits': bits,
    }


def compute_transmitted_size_kb(shape: tuple, keep_ratio: float = 0.1,
                                bits: int = 8) -> float:
    """
    Compute exact transmitted data size in KB for a given tensor shape
    and compression config. Used by AdaDC protocol for latency estimation.

    Wire format: sparse (index, value) pairs + per-sample metadata.
      - Each kept element: `bits` for value + 16 bits for index
      - Metadata: 2 × 32 bits (min/max for dequantization)

    Args:
        shape: (C, H, W) or (B, C, H, W) tensor shape
        keep_ratio: fraction of elements to keep
        bits: quantization bits
    Returns:
        size in KB (per sample)
    """
    if len(shape) == 4:
        _, C, H, W = shape
    else:
        C, H, W = shape
    total_elements = C * H * W
    kept_elements = max(1, int(total_elements * keep_ratio))
    compressed_bits = kept_elements * (bits + 16) + 2 * 32
    return compressed_bits / 8 / 1024


def make_compress_fn(keep_ratio: float = 0.1, bits: int = 8):
    """Factory function that returns a compression callable for the sampler."""
    def fn(delta_v):
        return compress_delta_v(delta_v, keep_ratio, bits)
    return fn


if __name__ == "__main__":
    # Quick test
    dv = torch.randn(4, 3, 32, 32)
    print("Original δv stats:")
    print(f"  shape: {dv.shape}, nonzero: {(dv != 0).sum().item()}")

    for kr in [0.05, 0.10, 0.20]:
        for bits in [4, 8]:
            compressed = compress_delta_v(dv, keep_ratio=kr, bits=bits)
            stats = compute_compression_ratio(dv, keep_ratio=kr, bits=bits)
            mse = ((compressed - dv) ** 2).mean().item()
            print(f"  keep={kr:.0%}, bits={bits}: "
                  f"ratio={stats['compression_ratio']:.3f} "
                  f"({stats['compressed_size_kb']:.2f} KB vs {stats['original_size_kb']:.2f} KB), "
                  f"MSE={mse:.4f}")
