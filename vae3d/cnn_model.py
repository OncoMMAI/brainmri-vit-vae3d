from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

def _gn(channels: int, groups: int=32) -> nn.GroupNorm:
    g = min(groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels, eps=1e-06, affine=True)

class ResBlock3D(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, groups: int=32, dropout: float=0.0):
        super().__init__()
        self.norm1 = _gn(in_ch, groups)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = _gn(out_ch, groups)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)

class AttnBlock3D(nn.Module):

    def __init__(self, channels: int, num_heads: int=1, groups: int=32):
        super().__init__()
        if channels % num_heads != 0:
            for h in (8, 4, 2, 1):
                if channels % h == 0:
                    num_heads = h
                    break
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = _gn(channels, groups)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, D * H * W)
        q, k, v = qkv.unbind(1)
        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(B, C, D, H, W)
        return x + self.proj(out)

class Downsample3D(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample3D(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        out_per_ch = B * (D * 2) * (H * 2) * (W * 2)
        max_chunk_c = max(1, 2 ** 30 // max(1, out_per_ch))
        if max_chunk_c >= C:
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        else:
            chunks = [F.interpolate(x[:, i:i + max_chunk_c], scale_factor=2.0, mode='nearest') for i in range(0, C, max_chunk_c)]
            x = torch.cat(chunks, dim=1)
        return self.conv(x)

class Encoder3D(nn.Module):

    def __init__(self, in_channels: int, base_channels: int, channel_mults: Sequence[int], latent_channels: int, num_res_blocks: int, groups: int, dropout: float, use_checkpoint: bool):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        chs = [base_channels * m for m in channel_mults]
        self.in_conv = nn.Conv3d(in_channels, chs[0], 3, padding=1)
        self.down = nn.ModuleList()
        in_ch = chs[0]
        for level, out_ch in enumerate(chs):
            blocks = nn.ModuleList([ResBlock3D(in_ch if i == 0 else out_ch, out_ch, groups, dropout) for i in range(num_res_blocks)])
            downsample = Downsample3D(out_ch) if level < len(chs) - 1 else nn.Identity()
            self.down.append(nn.ModuleDict({'blocks': blocks, 'downsample': downsample}))
            in_ch = out_ch
        self.mid_block1 = ResBlock3D(in_ch, in_ch, groups, dropout)
        self.mid_attn = AttnBlock3D(in_ch, groups=groups)
        self.mid_block2 = ResBlock3D(in_ch, in_ch, groups, dropout)
        self.out_norm = _gn(in_ch, groups)
        self.out_conv = nn.Conv3d(in_ch, 2 * latent_channels, 3, padding=1)

    def _maybe_ckpt(self, fn, x):
        if self.use_checkpoint and self.training and x.requires_grad:
            return ckpt.checkpoint(fn, x, use_reentrant=False)
        return fn(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        for level in self.down:
            for block in level['blocks']:
                x = self._maybe_ckpt(block, x)
            x = level['downsample'](x)
        x = self._maybe_ckpt(self.mid_block1, x)
        x = self.mid_attn(x)
        x = self._maybe_ckpt(self.mid_block2, x)
        x = self.out_conv(F.silu(self.out_norm(x)))
        return x

class Decoder3D(nn.Module):

    def __init__(self, out_channels: int, base_channels: int, channel_mults: Sequence[int], latent_channels: int, num_res_blocks: int, groups: int, dropout: float, use_checkpoint: bool):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        chs = [base_channels * m for m in channel_mults]
        self.in_conv = nn.Conv3d(latent_channels, chs[-1], 3, padding=1)
        self.mid_block1 = ResBlock3D(chs[-1], chs[-1], groups, dropout)
        self.mid_attn = AttnBlock3D(chs[-1], groups=groups)
        self.mid_block2 = ResBlock3D(chs[-1], chs[-1], groups, dropout)
        self.up = nn.ModuleList()
        in_ch = chs[-1]
        for level, out_ch in enumerate(reversed(chs)):
            blocks = nn.ModuleList([ResBlock3D(in_ch if i == 0 else out_ch, out_ch, groups, dropout) for i in range(num_res_blocks)])
            upsample = Upsample3D(out_ch) if level < len(chs) - 1 else nn.Identity()
            self.up.append(nn.ModuleDict({'blocks': blocks, 'upsample': upsample}))
            in_ch = out_ch
        self.out_norm = _gn(in_ch, groups)
        self.out_conv = nn.Conv3d(in_ch, out_channels, 3, padding=1)

    def _maybe_ckpt(self, fn, x):
        if self.use_checkpoint and self.training and x.requires_grad:
            return ckpt.checkpoint(fn, x, use_reentrant=False)
        return fn(x)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(z)
        x = self._maybe_ckpt(self.mid_block1, x)
        x = self.mid_attn(x)
        x = self._maybe_ckpt(self.mid_block2, x)
        for level in self.up:
            for block in level['blocks']:
                x = self._maybe_ckpt(block, x)
            x = level['upsample'](x)
        return self.out_conv(F.silu(self.out_norm(x)))

    def get_last_layer(self) -> torch.Tensor:
        return self.out_conv.weight

@dataclass
class DiagonalGaussianDistribution:
    mean: torch.Tensor
    logvar: torch.Tensor

    def __post_init__(self):
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        return 0.5 * (self.mean.pow(2) + self.var - 1.0 - self.logvar).mean()

@dataclass
class CNNVAE3DConfig:
    """Architecture fields stored in a released CNN checkpoint ``model_config``."""
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 128
    channel_mults: Sequence[int] = (1, 2, 4, 4)
    latent_channels: int = 16
    num_res_blocks: int = 2
    num_res_blocks_decoder: Optional[int] = None
    norm_groups: int = 32
    dropout: float = 0.0
    use_checkpoint: bool = False
    patch_size: tuple = (8, 8, 8)

    def __post_init__(self):
        self.channel_mults = tuple(int(x) for x in self.channel_mults)
        self.patch_size = tuple(int(x) for x in self.patch_size)
        expected = 2 ** (len(self.channel_mults) - 1)
        if self.patch_size != (expected, expected, expected):
            raise ValueError(f"patch_size={self.patch_size} does not match channel_mults "
                             f"spatial compression {expected}")


class CNNVAE3D(nn.Module):
    kind = "cnn"

    def __init__(self, config: Optional[CNNVAE3DConfig]=None, in_channels: int=1, out_channels: int=1, base_channels: int=128, channel_mults: Sequence[int]=(1, 2, 4, 4), latent_channels: int=64, num_res_blocks: int=2, num_res_blocks_decoder: Optional[int]=None, norm_groups: int=32, dropout: float=0.0, use_checkpoint: bool=False, patch_size: tuple=(8, 8, 8)):
        super().__init__()
        if config is not None:
            in_channels = config.in_channels
            out_channels = config.out_channels
            base_channels = config.base_channels
            channel_mults = config.channel_mults
            latent_channels = config.latent_channels
            num_res_blocks = config.num_res_blocks
            num_res_blocks_decoder = config.num_res_blocks_decoder
            norm_groups = config.norm_groups
            dropout = config.dropout
            use_checkpoint = config.use_checkpoint
            patch_size = config.patch_size
        if num_res_blocks_decoder is None:
            num_res_blocks_decoder = num_res_blocks + 1
        self.config = config or CNNVAE3DConfig(
            in_channels=in_channels, out_channels=out_channels,
            base_channels=base_channels, channel_mults=channel_mults,
            latent_channels=latent_channels, num_res_blocks=num_res_blocks,
            num_res_blocks_decoder=num_res_blocks_decoder,
            norm_groups=norm_groups, dropout=dropout,
            use_checkpoint=use_checkpoint, patch_size=patch_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_downs = len(self.config.channel_mults) - 1
        self.spatial_compression = 2 ** self.num_downs
        self.patch_size = self.config.patch_size
        self.encoder = Encoder3D(in_channels=in_channels, base_channels=base_channels, channel_mults=channel_mults, latent_channels=latent_channels, num_res_blocks=num_res_blocks, groups=norm_groups, dropout=dropout, use_checkpoint=use_checkpoint)
        self.decoder = Decoder3D(out_channels=out_channels, base_channels=base_channels, channel_mults=channel_mults, latent_channels=latent_channels, num_res_blocks=num_res_blocks_decoder, groups=norm_groups, dropout=dropout, use_checkpoint=use_checkpoint)

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        return DiagonalGaussianDistribution(mean, logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, sample: bool=True):
        posterior = self.encode(x)
        z = posterior.sample() if sample else posterior.mode()
        recon = self.decode(z)
        return (recon, posterior)

    def get_last_layer(self) -> torch.Tensor:
        return self.decoder.get_last_layer()

def build_vae(cfg: dict) -> CNNVAE3D:
    vcfg = cfg.get('vae', {})
    nrb_dec = vcfg.get('num_res_blocks_decoder', None)
    return CNNVAE3D(in_channels=int(vcfg.get('in_channels', 1)), out_channels=int(vcfg.get('out_channels', 1)), base_channels=int(vcfg.get('base_channels', 128)), channel_mults=tuple(vcfg.get('channel_mults', [1, 2, 4, 4])), latent_channels=int(vcfg.get('latent_channels', 64)), num_res_blocks=int(vcfg.get('num_res_blocks', 2)), num_res_blocks_decoder=int(nrb_dec) if nrb_dec is not None else None, norm_groups=int(vcfg.get('norm_groups', 32)), dropout=float(vcfg.get('dropout', 0.0)), use_checkpoint=bool(vcfg.get('use_checkpoint', False)))

def load_vae_state_dict(vae: CNNVAE3D, state: dict, key: str='vae', verbose: bool=True):
    sd = state.get(key) or state.get('vae') or state.get('ema') or state
    if any((k.startswith('_orig_mod.') for k in sd.keys())):
        sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
        if verbose:
            print('[vae load] stripped _orig_mod. prefix (torch.compile ckpt)')
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    if verbose:
        if missing or unexpected:
            print(f'[vae load] missing={len(missing)} unexpected={len(unexpected)}')
        else:
            print('[vae load] clean (all keys matched)')
    if len(missing) > 10 or len(unexpected) > 10:
        raise RuntimeError(f"VAE state-dict mismatch: missing={len(missing)} unexpected={len(unexpected)}. The model would silently run on random weights and produce noise. Check arch in cfg.vae matches the trained checkpoint, and that the right ckpt key ('vae' or 'ema') is selected via cfg.vae_ckpt_key.")
    return (missing, unexpected)
