"""Safe weight loading and whole-volume inference with explicit latent geometry.

The released models are trained on ``[0, 1]`` inputs with a linear pixel head.
No [-1, 1] remapping is applied.  Both the self-describing training checkpoint
(``.pt`` with ``model_config`` and ``ema``) and an exported safetensors file
(plus a JSON config) are supported.
"""

from contextlib import nullcontext
from dataclasses import dataclass, fields
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from .model import VAE3DConfig, ViTVAE3D
from .cnn_model import CNNVAE3D, CNNVAE3DConfig


def _resolve_weights(weights):
    weights = Path(weights)
    if weights.suffix == ".path":
        locator = weights.read_text().strip()
        if not locator or "\n" in locator:
            raise ValueError("Weight locator must contain exactly one path")
        located = Path(locator).expanduser()
        weights = located if located.is_absolute() else weights.parent / located
    if not weights.is_file():
        raise FileNotFoundError(weights)
    return weights


def _config_from_dict(raw):
    raw = dict(raw)
    if "patch_size" in raw:
        raw["patch_size"] = tuple(raw["patch_size"])
    if "channel_mults" in raw:
        raw["channel_mults"] = tuple(raw["channel_mults"])
    if "base_channels" in raw or "channel_mults" in raw:
        allowed = {f.name for f in fields(CNNVAE3DConfig)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown CNN architecture fields: {sorted(unknown)}")
        return "cnn", CNNVAE3DConfig(**raw)
    allowed = {f.name for f in fields(VAE3DConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown ViT architecture fields: {sorted(unknown)}")
    return "vitok", VAE3DConfig(**raw)


def _load_config_file(config):
    raw = json.loads(Path(config).read_text())
    return _config_from_dict(raw)


def _build_model(kind, cfg):
    if kind == "cnn":
        return CNNVAE3D(
            in_channels=cfg.in_channels, out_channels=cfg.out_channels,
            base_channels=cfg.base_channels, channel_mults=cfg.channel_mults,
            latent_channels=cfg.latent_channels, num_res_blocks=cfg.num_res_blocks,
            num_res_blocks_decoder=cfg.num_res_blocks_decoder,
            norm_groups=cfg.norm_groups, dropout=cfg.dropout,
            use_checkpoint=cfg.use_checkpoint, patch_size=cfg.patch_size)
    return ViTVAE3D(cfg)


def load_model(weights, config=None, device="cpu", variant="ema"):
    """Load a released model.

    Parameters
    ----------
    weights:
        ``.safetensors``, a self-describing training ``.pt`` checkpoint, or a
        ``.path`` locator file.
    config:
        JSON architecture file.  Required for safetensors; optional for a
        ``.pt`` checkpoint because the checkpoint already contains
        ``model_config``.
    variant:
        For ``.pt`` checkpoints, select ``"ema"`` (default, the released
        inference weights) or ``"model"``.
    """
    weights = _resolve_weights(weights)
    if variant not in ("ema", "model"):
        raise ValueError("variant must be 'ema' or 'model'")

    if weights.suffix == ".safetensors":
        if config is None:
            raise ValueError("A JSON config is required when loading safetensors")
        kind, cfg = _load_config_file(config)
        state = load_file(str(weights), device="cpu")
    else:
        if weights.suffix not in (".pt", ".pth", ".ckpt"):
            raise ValueError("Expected a .safetensors, .pt/.pth/.ckpt checkpoint, or a .path locator")
        payload = torch.load(weights, map_location="cpu", mmap=True, weights_only=False)
        if not isinstance(payload, dict) or "model_config" not in payload:
            raise ValueError("Checkpoint is missing model_config")
        if config is not None:
            kind, cfg = _load_config_file(config)
        else:
            kind, cfg = _config_from_dict(payload["model_config"])
        state = payload.get(variant)
        if state is None and variant == "ema" and payload.get("model") is not None:
            # A checkpoint without an EMA entry is usable only as its live model.
            state = payload["model"]
        if state is None:
            raise ValueError(f"Checkpoint has no {variant!r} state dict")
        del payload

    model = _build_model(kind, cfg)
    model.load_state_dict(state, strict=True)
    model.kind = kind
    return model.eval().requires_grad_(False).to(device)


@dataclass(frozen=True)
class VolumeGeometry:
    original_shape: tuple
    padded_shape: tuple
    pad_before: tuple


def pad_volume(x, patch):
    if x.ndim != 5 or any(s < 1 for s in x.shape):
        raise ValueError("Expected nonempty (B,C,D,H,W) input")
    shape = tuple(x.shape[-3:])
    total = tuple((-s) % p for s, p in zip(shape, patch))
    before = tuple(t // 2 for t in total)
    pads = tuple(v for b, t in reversed(list(zip(before, total))) for v in (b, t - b))
    padded = F.pad(x, pads, value=0.0)
    return padded, VolumeGeometry(shape, tuple(padded.shape[-3:]), before)


class VAEInference:
    def __init__(self, model, precision="bf16"):
        if model.config.in_channels != 1:
            raise ValueError("Volume inference requires a single-channel model")
        if precision not in ("fp32", "bf16", "fp16"):
            raise ValueError("precision must be fp32, bf16, or fp16")
        self.model = model.eval().requires_grad_(False)
        self.device = next(model.parameters()).device
        self.precision = precision
        if self.device.type != "cuda" and precision != "fp32":
            raise ValueError("CPU inference requires precision='fp32'")
        if precision == "bf16" and self.device.type == "cuda":
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("GPU does not support bf16; choose fp32 or fp16")

    def _autocast(self):
        if self.precision == "fp32":
            return nullcontext()
        return torch.autocast("cuda", dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[self.precision])

    @torch.inference_mode()
    def encode(self, image01, sample=False):
        """Encode ``(B,1,D,H,W)`` values in [0,1] to ``(B,C,Dz,Hy,Wx)`` latents.

        No latent scaling or shift is applied.  The posterior mean is returned
        by default; pass ``sample=True`` for a stochastic draw.
        """
        x = torch.as_tensor(image01, dtype=torch.float32, device=self.device)
        if x.ndim != 5 or x.shape[1] != self.model.config.in_channels:
            raise ValueError("Expected (B,1,D,H,W) input")
        if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError("Input must be finite and normalized to [0,1]")
        work, geometry = pad_volume(x, self.model.config.patch_size)
        with self._autocast():
            if self.model.kind == "cnn":
                posterior = self.model.encode(work)
                z = posterior.sample() if sample else posterior.mode()
                grid = tuple(z.shape[-3:])
            else:
                posterior, grid = self.model.encode(work)
                z = posterior.sample() if sample else posterior.mode()
                z = z.transpose(1, 2).reshape(x.shape[0], self.model.config.latent_channels, *grid)
        return z, geometry

    @torch.inference_mode()
    def decode(self, latents, geometry=None):
        """Decode ``(B,C,Dz,Hy,Wx)`` latents to float32 [0,1] images."""
        z = torch.as_tensor(latents, device=self.device, dtype=torch.float32)
        cfg = self.model.config
        if z.ndim != 5 or z.shape[1] != cfg.latent_channels or any(n < 1 for n in z.shape):
            raise ValueError("Expected nonempty (B,C,Dz,Hy,Wx) latent tensor")
        if not torch.isfinite(z).all():
            raise ValueError("Latents must be finite")
        grid = tuple(z.shape[-3:])
        target = tuple(g * p for g, p in zip(grid, cfg.patch_size))
        if geometry is not None:
            if target != geometry.padded_shape or len(geometry.original_shape) != 3 or len(geometry.pad_before) != 3:
                raise ValueError("Geometry does not match latent grid")
            if any(s <= 0 or b < 0 or b + s > t for s, b, t in zip(geometry.original_shape,
                                                                    geometry.pad_before, target)):
                raise ValueError("Invalid crop geometry")
        with self._autocast():
            if self.model.kind == "cnn":
                result = self.model.decode(z)
            else:
                tokens = z.flatten(2).transpose(1, 2).contiguous()
                result = self.model.decode(tokens, grid)
        result = result.float().clamp(0, 1)
        if geometry is not None:
            crop = tuple(slice(b, b + s) for b, s in zip(geometry.pad_before, geometry.original_shape))
            result = result[(slice(None), slice(None), *crop)]
        return result

    def reconstruct(self, array01):
        array = np.asarray(array01, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError("Expected one (D,H,W) volume")
        z, geometry = self.encode(array[None, None])
        result = self.decode(z, geometry)[0, 0].cpu().numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite reconstruction; retry with fp32 precision")
        return result
