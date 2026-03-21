"""
models.py — Cloud (full) and Edge (compressed) UNet for Rectified Flow on CIFAR-10.

Cloud model: standard UNet with ~35M params
Edge model:  pruned UNet with ~4M params (simulating on-device constraints)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Building blocks
# ============================================================

def num_groups(ch: int, max_groups: int = 32) -> int:
    """Largest divisor of ch that is <= max_groups."""
    for g in range(min(max_groups, ch), 0, -1):
        if ch % g == 0:
            return g
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for the time variable t ∈ [0,1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with time-conditioning via FiLM (scale + shift)."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(num_groups(in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2),  # scale and shift
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(num_groups(out_ch), out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        # FiLM conditioning
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class SelfAttention(nn.Module):
    """Simple single-head self-attention for spatial features."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups(ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax(q.transpose(-1, -2) @ k / math.sqrt(C), dim=-1)
        out = (v @ attn.transpose(-1, -2)).reshape(B, C, H, W)
        return x + self.proj(out)


# ============================================================
# UNet
# ============================================================

class UNet(nn.Module):
    """
    Configurable UNet for velocity prediction v(x_t, t).

    Args:
        in_ch:       input channels (3 for CIFAR-10 RGB)
        base_ch:     base channel width
        ch_mults:    channel multipliers per resolution level
        num_res:     number of ResBlocks per level
        attn_resolutions: set of downsampled spatial sizes where attention is applied
        dropout:     dropout rate
    """

    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 128,
        ch_mults: tuple = (1, 2, 2, 2),
        num_res: int = 2,
        attn_resolutions: tuple = (16,),
        dropout: float = 0.1,
    ):
        super().__init__()
        time_dim = base_ch * 4

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Initial projection
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Encoder
        self.encoder = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        channels = [base_ch]
        ch = base_ch
        current_res = 32  # CIFAR-10 is 32x32

        for level, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            for _ in range(num_res):
                block = ResBlock(ch, out_ch, time_dim, dropout)
                self.encoder.append(block)
                ch = out_ch
                channels.append(ch)
            if level != len(ch_mults) - 1:
                self.downsamplers.append(Downsample(ch))
                channels.append(ch)
                current_res //= 2
            else:
                self.downsamplers.append(nn.Identity())

        # Middle
        self.mid_block1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn = SelfAttention(ch)
        self.mid_block2 = ResBlock(ch, ch, time_dim, dropout)

        # Decoder
        self.decoder = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for level, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base_ch * mult
            for i in range(num_res + 1):
                skip_ch = channels.pop()
                block = ResBlock(ch + skip_ch, out_ch, time_dim, dropout)
                self.decoder.append(block)
                ch = out_ch
            if level != 0:
                self.upsamplers.append(Upsample(ch))
            else:
                self.upsamplers.append(nn.Identity())

        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups(ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity field v(x_t, t).

        Args:
            x: noisy input [B, C, H, W]
            t: time variable [B] in [0, 1]
        Returns:
            v: predicted velocity [B, C, H, W]
        """
        t_emb = self.time_embed(t)
        h = self.init_conv(x)

        # Encoder with skip connections
        skips = [h]
        enc_idx = 0
        down_idx = 0
        for level, mult in enumerate(self._ch_mults()):
            for _ in range(self._num_res()):
                h = self.encoder[enc_idx](h, t_emb)
                skips.append(h)
                enc_idx += 1
            if level != len(self._ch_mults()) - 1:
                h = self.downsamplers[down_idx](h)
                skips.append(h)
            down_idx += 1

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder
        dec_idx = 0
        up_idx = 0
        for level, mult in reversed(list(enumerate(self._ch_mults()))):
            for i in range(self._num_res() + 1):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.decoder[dec_idx](h, t_emb)
                dec_idx += 1
            if level != 0:
                h = self.upsamplers[up_idx](h)
            up_idx += 1

        return self.out(h)

    def _ch_mults(self):
        # Extract from init params — store during init for cleanliness
        # For simplicity, we re-derive from encoder structure
        # This is a helper; in practice, store as attribute
        return getattr(self, '_stored_ch_mults', (1, 2, 2, 2))

    def _num_res(self):
        return getattr(self, '_stored_num_res', 2)


# ============================================================
# Factory functions
# ============================================================

def build_cloud_model() -> UNet:
    """
    Cloud model: full-size UNet (~35M params).
    base_ch=128, ch_mults=(1,2,2,2), num_res=2, attention at 16x16.
    """
    model = UNet(
        in_ch=3,
        base_ch=128,
        ch_mults=(1, 2, 2, 2),
        num_res=2,
        attn_resolutions=(16,),
        dropout=0.1,
    )
    model._stored_ch_mults = (1, 2, 2, 2)
    model._stored_num_res = 2
    return model


def build_edge_model() -> UNet:
    """
    Edge model: pruned UNet (~4M params).
    base_ch=48, ch_mults=(1,2,2), num_res=1, no attention.
    Simulates a model that could run on a mobile device.
    """
    model = UNet(
        in_ch=3,
        base_ch=48,
        ch_mults=(1, 2, 2),
        num_res=1,
        attn_resolutions=(),  # no attention for efficiency
        dropout=0.0,
    )
    model._stored_ch_mults = (1, 2, 2)
    model._stored_num_res = 1
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    cloud = build_cloud_model()
    edge = build_edge_model()
    print(f"Cloud model: {count_params(cloud) / 1e6:.1f}M params")
    print(f"Edge model:  {count_params(edge) / 1e6:.1f}M params")

    # Quick forward pass test
    x = torch.randn(2, 3, 32, 32)
    t = torch.rand(2)
    v_cloud = cloud(x, t)
    v_edge = edge(x, t)
    print(f"Cloud output shape: {v_cloud.shape}")
    print(f"Edge output shape:  {v_edge.shape}")
