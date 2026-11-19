"""Export a trusted training checkpoint to exact float32 inference tensors.

The released ViT-VAE and CNN checkpoints are self-describing (``model_config`` plus
``model``/``ema`` state dicts).  This script writes only the selected state dict
to safetensors, together with the architecture JSON and export metadata.
"""
import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from vae3d.model import VAE3DConfig, ViTVAE3D
from vae3d.cnn_model import CNNVAE3D, CNNVAE3DConfig


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--variant", choices=["ema", "model"], default="ema")
    p.add_argument("--expected-step", type=int, default=None)
    p.add_argument("--expected-parameters", type=int, default=None)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "model.safetensors"
    partial = args.output / ".model.safetensors.partial"
    if any((args.output / name).exists() for name in
           ["model.safetensors", ".model.safetensors.partial", "config.json", "weights_info.json"]):
        raise FileExistsError("Export destination contains artifacts; choose an empty directory")

    payload = torch.load(args.source, map_location="cpu", mmap=True, weights_only=False)
    arch = payload.get("arch")
    if arch not in (None, "vitok", "conflux"):
        raise ValueError(f"Unsupported checkpoint architecture: {arch!r}")
    if args.expected_step is not None and payload.get("step") != args.expected_step:
        raise ValueError(f"Checkpoint step {payload.get('step')} != expected {args.expected_step}")
    saved = payload.get("model_config")
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint is missing model_config")
    is_cnn = arch == "conflux" or "base_channels" in saved
    if is_cnn:
        names = {f.name for f in fields(CNNVAE3DConfig)}
        unknown = set(saved) - names
        if unknown:
            raise ValueError(f"Unknown CNN checkpoint architecture fields: {sorted(unknown)}")
        cfg = CNNVAE3DConfig(**saved)
    else:
        names = {f.name for f in fields(VAE3DConfig)}
        unknown = set(saved) - names
        if unknown:
            raise ValueError(f"Unknown ViT checkpoint architecture fields: {sorted(unknown)}")
        cfg = VAE3DConfig(**saved)
    state = payload.get(args.variant)
    if state is None:
        raise ValueError(f"Checkpoint has no {args.variant!r} state dict")
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) and v.dtype == torch.float32
                                              for v in state.values()):
        raise ValueError("Expected a float32 state dictionary")

    with torch.device("meta"):
        if is_cnn:
            model = CNNVAE3D(
                in_channels=cfg.in_channels, out_channels=cfg.out_channels,
                base_channels=cfg.base_channels, channel_mults=cfg.channel_mults,
                latent_channels=cfg.latent_channels, num_res_blocks=cfg.num_res_blocks,
                num_res_blocks_decoder=cfg.num_res_blocks_decoder,
                norm_groups=cfg.norm_groups, dropout=cfg.dropout,
                use_checkpoint=cfg.use_checkpoint, patch_size=cfg.patch_size)
        else:
            model = ViTVAE3D(cfg)
    model.load_state_dict(state, strict=True, assign=True)
    parameter_count = sum(v.numel() for v in model.parameters())
    if args.expected_parameters is not None and parameter_count != args.expected_parameters:
        raise ValueError(f"Unexpected parameter count: {parameter_count}")
    print(f"validated architecture ({parameter_count:,} parameters); exporting {args.variant} tensors only",
          flush=True)

    save_file({k: v.detach().contiguous() for k, v in state.items()}, str(partial))
    with safe_open(str(partial), framework="pt", device="cpu") as f:
        if set(f.keys()) != set(state):
            raise ValueError("Export key mismatch")
        for k, v in state.items():
            if not torch.equal(f.get_tensor(k), v):
                raise ValueError(f"Export tensor mismatch: {k}")
    partial.replace(target)
    (args.output / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    info = {
        "format": "safetensors",
        "dtype": "float32",
        "parameters": parameter_count,
        "tensor_count": len(state),
        "size_bytes": target.stat().st_size,
        "sha256": sha256(target),
        "variant": args.variant,
        "arch": "conflux" if is_cnn else "vitok",
        "source_step": payload.get("step"),
        "all_tensors_equal_to_source": True,
        "contains_optimizer_or_discriminator": False,
    }
    (args.output / "weights_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
