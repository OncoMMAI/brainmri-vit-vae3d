"""Standalone ViTok-v2-style 3D VAE used by the released 500M/250M models.

The implementation is a self-contained copy of the architecture used for
training, with no imports from the training project.  The checkpoint payloads
written by training store the architecture under ``model_config`` and the
inference weights under ``ema`` (default) or ``model``.
"""

from dataclasses import dataclass
from math import prod
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class VAE3DConfig:
    """Architecture fields stored in ``payload['model_config']``."""

    spatial_dims: int = 3
    in_channels: int = 1
    patch_size: Tuple[int, ...] = (8, 8, 8)
    latent_channels: int = 16
    width: int = 1024
    heads: int = 16
    encoder_depth: int = 4
    decoder_depth: int = 36
    ffn_dim: int = 2688
    variational: bool = True
    gradient_checkpointing: bool = False
    layer_scale_init: float = 1e-4

    def __post_init__(self):
        self.patch_size = tuple(self.patch_size)
        if self.spatial_dims not in (2, 3) or len(self.patch_size) != self.spatial_dims:
            raise ValueError("patch_size must match spatial_dims (2 or 3)")
        if any(p <= 0 for p in self.patch_size):
            raise ValueError("patch sizes must be positive")
        if min(self.in_channels, self.latent_channels, self.width, self.heads,
               self.encoder_depth, self.decoder_depth, self.ffn_dim) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        head_dim = self.width // self.heads
        if head_dim < 2 * self.spatial_dims or head_dim % 2:
            raise ValueError("head dimension must be even and provide a rotary pair per spatial axis")


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), 1e-6).to(x.dtype)


def rope(x, coords):
    """Axial RoPE; the 3D extension distributes rotary pairs over D, H, W.

    The order is the same as in training: split ``head_dim/2`` rotary pairs
    across the spatial axes, then apply each axis with its own coordinates.
    """
    pairs = x.shape[-1] // 2
    dims = coords.shape[-1]
    sizes = [pairs // dims + (i < pairs % dims) for i in range(dims)]
    angles = []
    for axis, n in enumerate(sizes):
        inv = 10000.0 ** (-torch.arange(n, device=x.device, dtype=torch.float32) / n)
        angles.append(coords[:, axis:axis + 1].float() * inv)
    angle = torch.cat(angles, -1)[None, None]
    real, imag = x.float().reshape(*x.shape[:-1], pairs, 2).unbind(-1)
    y = torch.stack((real * angle.cos() - imag * angle.sin(),
                     real * angle.sin() + imag * angle.cos()), -1).flatten(-2)
    return y.to(x.dtype)


class Block(nn.Module):
    """Pre-norm parallel attention + SwiGLU block used by ViTok-v2."""

    def __init__(self, c: VAE3DConfig):
        super().__init__()
        self.heads = c.heads
        self.norm = RMSNorm(c.width)
        self.qkv = nn.Linear(c.width, 3 * c.width, bias=False)
        self.proj = nn.Linear(c.width, c.width, bias=False)
        self.q_norm = RMSNorm(c.width // c.heads)
        self.k_norm = RMSNorm(c.width // c.heads)
        self.gate_value = nn.Linear(c.width, 2 * c.ffn_dim, bias=False)
        self.ffn_out = nn.Linear(c.ffn_dim, c.width, bias=False)
        self.scale = nn.Parameter(torch.full((c.width,), c.layer_scale_init))

    def forward(self, x, coords):
        h = self.norm(x)
        b, n, w = h.shape
        q, k, v = self.qkv(h).reshape(b, n, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = rope(self.q_norm(q), coords), rope(self.k_norm(k), coords)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, w)
        value, gate = self.gate_value(h).chunk(2, -1)
        return x + self.scale * (self.proj(attn) + self.ffn_out(value * F.silu(gate)))


class DiagonalGaussian:
    def __init__(self, moments):
        self.mean, logvar = moments.chunk(2, -1)
        self.logvar = logvar.float().clamp(-30, 20)

    def sample(self):
        return self.mean + (torch.randn_like(self.mean.float()) * (0.5 * self.logvar).exp()).to(self.mean.dtype)

    def mode(self):
        return self.mean

    def kl(self):
        """Mean over tokens/channels per example; batch reduction belongs to loss."""
        return 0.5 * (self.mean.float().square() + self.logvar.exp() - 1 - self.logvar).mean((1, 2))


class ViTVAE3D(nn.Module):
    kind = "vitok"
    """Single-channel 3D VAE with 8x patch tokenisation and a 16-d Gaussian latent.

    Inputs are ``[B, 1, D, H, W]`` in ``[0, 1]``.  Spatial dimensions must be
    divisible by ``patch_size``.  ``encode`` returns a diagonal Gaussian over
    ``(D/p, H/p, W/p)`` tokens; ``decode`` accepts those same tokens and an
    explicit grid and returns a linear (non-squashed) reconstruction in [0,1]
    after clamping by the caller.
    """

    def __init__(self, config: VAE3DConfig):
        super().__init__()
        self.config = c = config
        pixels = c.in_channels * prod(c.patch_size)
        self.patch_embed = nn.Linear(pixels, c.width)
        self.encoder = nn.ModuleList([Block(c) for _ in range(c.encoder_depth)])
        self.to_latent = nn.Linear(c.width, c.latent_channels * (2 if c.variational else 1))
        self.latent_norm = nn.Identity() if c.variational else nn.LayerNorm(c.latent_channels, eps=1e-6,
                                                                           elementwise_affine=False)
        self.decoder_embed = nn.Linear(c.latent_channels, c.width)
        self.decoder = nn.ModuleList([Block(c) for _ in range(c.decoder_depth)])
        self.to_pixels = nn.Linear(c.width, pixels)

    def patchify(self, x):
        c = self.config
        if x.ndim != c.spatial_dims + 2 or x.shape[1] != c.in_channels:
            raise ValueError("expected [B,C,H,W] or [B,C,D,H,W] matching config")
        spatial = tuple(x.shape[2:])
        if any(s % p for s, p in zip(spatial, c.patch_size)):
            raise ValueError("spatial dimensions must be divisible by patch_size; pad/crop explicitly")
        grid = tuple(s // p for s, p in zip(spatial, c.patch_size))
        shape = [x.shape[0], c.in_channels]
        for g, p in zip(grid, c.patch_size):
            shape.extend((g, p))
        n = c.spatial_dims
        order = [0] + list(range(2, 2 + 2 * n, 2)) + [1] + list(range(3, 2 + 2 * n, 2))
        return x.reshape(shape).permute(order).reshape(x.shape[0], prod(grid), -1), grid

    def unpatchify(self, patches, grid):
        c = self.config
        n = c.spatial_dims
        x = patches.reshape(patches.shape[0], *grid, c.in_channels, *c.patch_size)
        order = [0, n + 1]
        for i in range(n):
            order.extend((1 + i, n + 2 + i))
        return x.permute(order).reshape(patches.shape[0], c.in_channels,
                                        *[g * p for g, p in zip(grid, c.patch_size)])

    def _coords(self, grid, device):
        return torch.stack(torch.meshgrid(*[torch.arange(g, device=device) for g in grid],
                                          indexing="ij"), -1).reshape(-1, len(grid))

    def _blocks(self, x, blocks, coords):
        for block in blocks:
            if self.training and self.config.gradient_checkpointing and torch.is_grad_enabled():
                x = checkpoint(block, x, coords, use_reentrant=False)
            else:
                x = block(x, coords)
        return x

    def encode(self, x):
        patches, grid = self.patchify(x)
        h = self._blocks(self.patch_embed(patches), self.encoder, self._coords(grid, x.device))
        latent = self.to_latent(h)
        return (DiagonalGaussian(latent) if self.config.variational else self.latent_norm(latent)), grid

    def decode(self, z, grid):
        if len(grid) != self.config.spatial_dims or z.ndim != 3 or z.shape[1] != prod(grid):
            raise ValueError("z must be [B,prod(grid),latent_channels] and grid must match spatial_dims")
        h = self._blocks(self.decoder_embed(z), self.decoder, self._coords(grid, z.device))
        return self.unpatchify(self.to_pixels(h), grid)

    def forward(self, x, sample_posterior=True):
        posterior, grid = self.encode(x)
        if self.config.variational:
            z = posterior.sample() if sample_posterior else posterior.mode()
            kl = posterior.kl().mean()
        else:
            z, kl = posterior, x.new_zeros((), dtype=torch.float32)
        return {"reconstruction": self.decode(z, grid), "z": z, "kl": kl, "grid": grid}


def build_model(config_path):
    """Build a model from a JSON architecture file (used by export/tests)."""
    import json
    with open(config_path) as f:
        return ViTVAE3D(VAE3DConfig(**json.load(f)))
