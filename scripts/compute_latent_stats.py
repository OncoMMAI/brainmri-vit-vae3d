"""Compute frozen per-channel posterior-mean latent statistics for a DiT.

This is the model-specific normalization hyperparameter: raw VAE latents are
not N(0,1), so a diffusion model needs a fixed `u = (z - mean) / std` transform.
The statistics must be estimated from a large training sample, using the
posterior mean on the same latent grid the DiT will read.

The script reads H5 `image/iso1mm` volumes (already 1 mm, [0,1]) or NIfTI files,
center-crops/pads them to the requested spatial size, encodes in batches, and
streams FP64 per-channel Welford/Chan statistics. It writes a JSON contract
compatible with the project's `LatentNormalizer` schema.

Example (1,000 training volumes, one GPU):

    python scripts/compute_latent_stats.py \
      --weights /path/to/vitok-500m-ema/model.safetensors \
      --config /path/to/vitok-500m-ema/config.json \
      --manifest data/records_train_spacing2.json \
      --h5-root /path/to/GBM_VLM_h5 \
      --max-items 1000 --subset-seed 20261119 \
      --spatial-size 192 192 144 \
      --out /path/to/vitok-500m-ema/latent_stats.json

`--max-items` is the number of volumes actually encoded; `0` means all entries
in the manifest (potentially very large). A subset is explicitly marked
`subset_estimate: true` in the output.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from vae3d import VAEInference, load_model, load_preprocess_config  # noqa: E402
from vae3d.io import prepare_nifti  # noqa: E402

SCHEMA_VERSION = 1
AXIS_ORDER = "C,D,H,W"


def sha256_file(path, chunk=1 << 23):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(d):
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stats_sha256(d):
    return hashlib.sha256(canonical_json({k: v for k, v in d.items()
                                          if k != "stats_sha256"}).encode()).hexdigest()


def center_crop_pad(arr, target, pad=0.0):
    out = np.full(target, pad, dtype=arr.dtype)
    src, dst = [], []
    for a, t in zip(arr.shape, target):
        if a >= t:
            s = (a - t) // 2
            src.append(slice(s, s + t))
            dst.append(slice(0, t))
        else:
            d = (t - a) // 2
            src.append(slice(0, a))
            dst.append(slice(d, d + a))
    out[tuple(dst)] = arr[tuple(src)]
    return out


def expand_manifest(raw):
    if isinstance(raw, dict):
        raw = raw.get("items", [])
    out = []
    for entry in raw:
        if isinstance(entry, str):
            out.append({"path": entry})
        elif isinstance(entry, dict) and "images" in entry:
            for image in entry.get("images", []):
                out.append({"path": image["path"], "modality": image.get("modality")})
        elif isinstance(entry, dict) and entry.get("path"):
            out.append(entry)
        else:
            raise ValueError(f"Unsupported manifest entry: {entry!r}")
    return out


def read_volume(entry, h5_root, spatial_size, mode, preprocess_config):
    path = Path(str(entry["path"]))
    if not path.is_absolute():
        path = Path(h5_root) / path
    if mode in ("h5", "h5_native"):
        import h5py
        key = "image/iso1mm" if mode == "h5" else "image/native"
        with h5py.File(path, "r") as f:
            if key not in f:
                raise KeyError(f"{path}: missing {key}")
            arr = np.asarray(f[key], dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[0]
    else:
        arr = prepare_nifti(path, "mri", config=preprocess_config).array01
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected one 3D volume, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{path}: nonfinite intensities")
    return center_crop_pad(arr.astype(np.float32, copy=False), spatial_size, 0.0)


def batch_stats(z):
    """z: (B,C,D,H,W) -> (n_positions, mean[C], M2[C]) in FP64."""
    x = np.asarray(z, dtype=np.float64)
    c = x.shape[1]
    x = x.reshape(c, -1)
    n = int(x.shape[1])
    mean = x.mean(axis=1)
    d = x - mean[:, None]
    return n, mean, (d * d).sum(axis=1)


def merge(a, b):
    n_a, m_a, M_a = a
    n_b, m_b, M_b = b
    if n_a == 0:
        return b
    if n_b == 0:
        return a
    n = n_a + n_b
    delta = m_b - m_a
    mean = m_a + delta * (n_b / n)
    M2 = M_a + M_b + delta * delta * (n_a * n_b / n)
    return int(n), mean, M2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--config", default=None,
                   help="architecture JSON; required for safetensors")
    p.add_argument("--variant", choices=["ema", "model"], default="ema")
    p.add_argument("--manifest", required=True)
    p.add_argument("--h5-root", default="")
    p.add_argument("--mode", choices=["h5", "h5_native", "nifti"], default="h5")
    p.add_argument("--spatial-size", nargs=3, type=int, default=[192, 192, 144])
    p.add_argument("--max-items", type=int, default=0,
                   help="number of volumes to encode; 0 = all manifest entries")
    p.add_argument("--subset-seed", type=int, default=20261119)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--preprocess-config", default=None)
    p.add_argument("--latent-group", default="latent")
    p.add_argument("--h5-key", default="latent/iso1mm/192x192x144/mean",
                   help="downstream latent-cache key recorded in the contract")
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    manifest_bytes = manifest_path.read_bytes()
    items = expand_manifest(json.loads(manifest_bytes))
    selected = items
    subset = 0 < args.max_items < len(items)
    if subset:
        rng = random.Random(args.subset_seed)
        selected = rng.sample(items, args.max_items)
    if not selected:
        raise SystemExit("No manifest entries selected")
    selection_sha = hashlib.sha256(
        "\n".join(str(x["path"]) for x in selected).encode()).hexdigest()
    duplicates = sum(c - 1 for c in collections.Counter(str(x["path"]) for x in selected).values() if c > 1)
    print(f"[latent-stats] manifest={len(items)} selected={len(selected)} "
          f"duplicates={duplicates} subset={subset}", flush=True)

    pp = load_preprocess_config(args.preprocess_config) if args.preprocess_config else None
    model = load_model(args.weights, args.config, args.device, args.variant)
    runner = VAEInference(model, args.precision)
    spatial = tuple(int(x) for x in args.spatial_size)
    if tuple(model.config.patch_size) != (8, 8, 8):
        raise SystemExit(f"expected 8x8x8 patch, got {model.config.patch_size}")

    acc = (0, None, None)
    modality_counts = collections.Counter()
    latent_shape = None
    n_ok = 0
    t0 = time.time()
    for start in range(0, len(selected), args.batch_size):
        chunk = selected[start:start + args.batch_size]
        arrays = []
        valid_entries = []
        for entry in chunk:
            try:
                arrays.append(read_volume(entry, args.h5_root, spatial, args.mode, pp))
                valid_entries.append(entry)
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {entry.get('path')}: {exc}", flush=True)
        if not arrays:
            continue
        x = torch.from_numpy(np.stack(arrays))[:, None].float().clamp(0, 1)
        with torch.inference_mode():
            z, _ = runner.encode(x)
        z = z.float().cpu().numpy()
        if latent_shape is None:
            latent_shape = tuple(int(v) for v in z.shape[1:])
        elif tuple(z.shape[1:]) != latent_shape:
            raise ValueError(f"latent shape changed: {z.shape[1:]} != {latent_shape}")
        acc = merge(acc, batch_stats(z))
        n_ok += len(arrays)
        for entry in valid_entries:
            modality_counts[str(entry.get("modality"))] += 1
        if n_ok % 100 == 0 or n_ok == len(selected):
            print(f"[latent-stats] {n_ok}/{len(selected)} "
                  f"{n_ok / max(time.time() - t0, 1e-9):.2f} vol/s", flush=True)
        del x, z

    if n_ok == 0:
        raise SystemExit("No volumes encoded")
    n_positions, mean, M2 = acc
    variance = np.maximum(M2 / n_positions, 0.0)
    std_raw = np.sqrt(variance)
    std_eff = np.maximum(std_raw, args.eps)
    stats = {
        "schema_version": SCHEMA_VERSION,
        "mode": "per_channel",
        "mean": [float(v) for v in mean],
        "std_raw": [float(v) for v in std_raw],
        "std_effective": [float(v) for v in std_eff],
        "eps": float(args.eps),
        "variance_ddof": 0,
        "floored_channels": sorted(int(i) for i in np.where(std_raw < args.eps)[0]),
        "n_entries": int(n_ok),
        "n_positions_per_channel": int(n_positions),
        "modality_counts": dict(sorted(modality_counts.items())),
        "weighting": "entry_uniform",
        "spatial_mask": "full_grid",
        "vae_checkpoint_path": str(args.weights),
        "vae_sha256": sha256_file(args.weights),
        "vae_config_sha256": sha256_file(args.config) if args.config else "embedded",
        "vae_arch": getattr(model, "kind", "unknown"),
        "cache_provenance": "direct posterior-mean encode; no latent cache",
        "manifest_path": str(args.manifest),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "selection_sha256": selection_sha,
        "subset_seed": args.subset_seed if subset else None,
        "subset_estimate": bool(subset),
        "latent_group": args.latent_group,
        "h5_key": args.h5_key,
        "source_image_key": "image/iso1mm" if args.mode == "h5" else args.mode,
        "posterior_mode": "mean",
        "shape": [int(v) for v in latent_shape],
        "axis_order": AXIS_ORDER,
        "spatial_size": list(spatial),
        "batch_size": int(args.batch_size),
        "precision": args.precision,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": "release/transformer-vae3d/scripts/compute_latent_stats.py",
        "n_duplicates": int(duplicates),
    }
    stats["stats_sha256"] = stats_sha256(stats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(stats, indent=2) + "\n")
    os.replace(tmp, args.out)
    print(f"[latent-stats] wrote {args.out}", flush=True)
    print(f"[latent-stats] n_entries={n_ok} shape={latent_shape} "
          f"stats_sha256={stats['stats_sha256']}", flush=True)
    with np.printoptions(precision=4, suppress=True):
        print(f"[latent-stats] mean={np.asarray(stats['mean'])}", flush=True)
        print(f"[latent-stats] std={np.asarray(stats['std_raw'])}", flush=True)


if __name__ == "__main__":
    main()
