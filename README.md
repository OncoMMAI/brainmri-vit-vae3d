# BrainMRI ViT-VAE 3D

High-compression variational autoencoders for 3D brain MRI and medical imaging.

Standalone inference code for the **ViT-VAE 500M**, **ViT-VAE 250M**, and
**CNN VAE** models used in *A High-Compression Transformer VAE for Efficient 3D
Medical Image Generation*.

All three models use an 8×8×8 patch and a 16-channel Gaussian latent per patch.
The ViT models use a shallow 4-block encoder with a deeper transformer decoder;
the CNN model is a 4-level convolutional UNet with two encoder and three decoder
residual blocks per level.

| model | parameters | architecture | latent | patch |
|---|---:|---|---:|---:|
| ViT-VAE 500M | 499,260,960 | ViT, 4/36 blocks, width 1024 | 16 | 8³ |
| ViT-VAE 250M | 249,197,600 | ViT, 4/16 blocks, width 1024 | 16 | 8³ |
| CNN VAE | 246,463,649 | conv UNet, base 128, 2/3 res-blocks | 16 | 8³ |

The input convention is **[0, 1]**, with a linear (non-squashed) pixel head.
There is no [-1, 1] remapping and no tanh output. The released weights are the
selected EMA inference weights.

This repository provides the standalone inference implementation. Training
data are not distributed.

## Installation

Use Python 3.10 or newer and install a CUDA-compatible PyTorch build for your
GPU, then:

```bash
python -m pip install -e '.[nifti]'
```

The tensor API requires only PyTorch, NumPy, and safetensors. The NIfTI command
additionally uses ANTsPy and SciPy. The validated environment uses PyTorch
2.5.1, NumPy 2.2.6, safetensors 0.7.0, ANTsPy 0.5.4, and SciPy 1.14.1. There are
no automatic model downloads or dependencies on the training project.

## Model files

Download the EMA inference weights and matching configs from Hugging Face.

| Model | Hugging Face |
|---|---|
| ViT-VAE 500M | [EMA weights](https://huggingface.co/NoobDoge/brainmri-vit-vae3d-500m-ema) |
| ViT-VAE 250M | [EMA weights](https://huggingface.co/NoobDoge/brainmri-vit-vae3d-250m-ema) |
| CNN VAE | [EMA weights](https://huggingface.co/NoobDoge/brainmri-vit-vae3d-cnn-246m-ema) |

Each model repo contains:

- `model.safetensors` — float32 EMA inference weights only.
- `config.json` — architecture.
- `weights_info.json` — parameter/tensor count, SHA-256, source step, export check.
- `preprocess.json` — frozen inference preprocessing.
- `latent_stats.json` — 1,000-volume per-channel latent statistics for DiT
  normalization (subset estimate; recompute on the full train split if needed).
- `README.md` — model card.

Download the 500M bundle with the Hugging Face CLI:

```bash
HF_HOME=/working/huggingface_cache hf download NoobDoge/brainmri-vit-vae3d-500m-ema
```

Use the returned snapshot directory for the inference examples below.
The CLI stores downloads under its configured Hugging Face cache.

The architecture configuration is also available locally at
`configs/vitok_500m.json`, `configs/vitok_250m.json`, `configs/cnn_246m.json`,
and as a 500M default in `configs/config.json`. The preprocessing hyperparameters
are in `configs/preprocess.json`.

## Preprocessing hyperparameters

The release uses the frozen evaluation pipeline. These values are also
machine-readable in `configs/preprocess.json`:

| field | value |
|---|---|
| orientation | RAS |
| target spacing | 1.0 × 1.0 × 1.0 mm |
| ANTs resampling | `use_voxels=false`, `interp_type=1` (nearest neighbor) |
| array order | DHW after ANTs XYZ→DHW transpose |
| intensity clip | whole-volume 0.5th–99.5th percentile |
| normalization | `(x - p0.5) / (p99.5 - p0.5 + 1e-8)`, then clamp to [0,1] |
| model input range | [0,1] |
| spatial padding | symmetric zero padding to a multiple of 8 per axis |
| CT padding repair | boundary-connected exact −3024 HU → −1000 HU, before resampling |
| latent normalization | **none** (no scale factor, shift, mean, or std) |
| output scaling | map back to the clipped source range by default |

`interp_type=1` is ANTs nearest-neighbor, retained deliberately because it is
what the evaluated benchmark used. The latent contract is raw posterior mean:
downstream diffusion models must apply their own latent normalization rather
than assuming N(0,1) or a Wan-style scale/shift.

## Reconstruct a NIfTI volume

```bash
python -m vae3d \
  --weights /path/to/model-snapshot/model.safetensors \
  --config /path/to/model-snapshot/config.json \
  --preprocess-config /path/to/model-snapshot/preprocess.json \
  --input input.nii.gz \
  --output outputs/reconstruction.nii.gz \
  --modality mri --device cuda:0 --precision bf16
```

For the 250M model, use the 250M bundle's `model.safetensors` and `config.json`.
For the CNN model, use the CNN bundle. For a self-describing raw training
checkpoint, pass the `.pt` file and omit `--config`; `--variant ema` (default)
selects the EMA weights and `--variant model` selects the live training
weights.

For CT, add `--modality ct`. Each input is one 3D, single-channel volume; process
different MRI sequences separately. CPU execution requires `--device cpu
--precision fp32` and can be slow. `--precision fp32` is also available on CUDA.
Existing output files are not overwritten.

Preprocessing follows the evaluated pipeline: ANTs RAS reorientation, 1 mm
resampling, and whole-volume 0.5/99.5 percentile clipping and normalization to
[0,1]. The existing resampling setting is **ANTs interpolation code 1 (nearest
neighbor)**; it is retained deliberately for numerical compatibility. Array axes
are transposed from ANTs XYZ to model DHW. The VAE receives [0,1] values, with
symmetric zero padding in [0,1] space to a multiple of eight. Reconstruction
uses the posterior mean and removes that padding. It does not silently crop or
tile the volume.

For CT, exact -3024 HU values connected to an image-array boundary are replaced
with -1000 HU **before resampling**. Isolated interior values are preserved. CT
inputs must already be in HU. Use `--no-ct-padding-repair` to disable this
specific padding convention. No generic low-intensity mask or anatomical
segmentation is applied.

The output is on the **preprocessed 1 mm grid**, retaining its origin, spacing,
and direction. It is not on the original acquisition grid. By default,
intensities are mapped back to the clipped source intensity range (HU for CT);
values removed by percentile clipping cannot be recovered. Use `--output-scale
normalized` for [0,1] output. Source images are never overwritten.

## Tensor and latent API

```python
import torch
from vae3d import load_model, VAEInference

model = load_model('/path/to/model-snapshot/model.safetensors',
                   '/path/to/model-snapshot/config.json', device='cuda:0')
runner = VAEInference(model, precision='bf16')
image01 = torch.rand(1, 1, 128, 128, 128, device='cuda:0')
z, geometry = runner.encode(image01)  # (1, 16, 16, 16, 16)
reconstruction01 = runner.decode(z, geometry)  # (1, 1, 128, 128, 128)
```

`encode` uses the posterior mean by default; `sample=True` samples from the
diagonal Gaussian. Latents have **no implicit scaling factor or shift**.
Downstream generators must establish their own latent normalization; for DiT
training, compute per-channel mean/std on the training split and apply
`u = (z - mean) / std`. Decode
externally generated 5D latents with `runner.decode(z)`; the result is
latent-grid size multiplied by the patch size. Retain `geometry` when recovering
an original shape that required padding.

The low-level API mirrors training. For the ViT models, `model.encode(x)`
returns a `DiagonalGaussian` over `(B, N, 16)` tokens plus the spatial grid, and
`model.decode(tokens, grid)` requires an explicit grid; `forward` returns a dict
with `reconstruction`, `z`, `kl`, and `grid`. For the CNN model, `encode(x)`
returns a 5D `DiagonalGaussian` and `decode(z)` takes the 5D latent directly;
`forward` returns `(reconstruction, posterior)`. No training loop, FSDP,
discriminator, or activation-checkpointing dependency is included.

Spatial-grid reduction is 512×; single-channel C16 scalar-element reduction is
32×. Neither quantity is a compressed file bitrate. Variable shapes are
supported within available memory. Full-volume inference can still run out of
memory for large inputs.

## Latent normalization for downstream DiT

The VAE boundary is intentionally raw: `runner.encode(...)` returns the
posterior mean, not a normalized latent. Before training a diffusion model,
compute per-channel statistics over a large training sample (posterior mean,
full latent grid) and freeze them:

```text
u[c] = (z[c] - mean[c]) / std[c]
z[c] = u[c] * std[c] + mean[c]
```

The release includes `scripts/compute_latent_stats.py` for this purpose. It
streams FP64 per-channel statistics from a manifest, records the VAE SHA-256,
manifest/selection hashes, sample count, and posterior policy, and writes a
JSON contract. Do not estimate the normalization from a single batch, and do
not reuse another VAE's statistics. The statistics are a model-specific
hyperparameter, not part of the VAE weights. The shipped `latent_stats.json`
files use 1,000 training volumes with a fixed seed and are marked
`subset_estimate: true`; rerun the script on the full training split for a
production DiT if the subset estimate is not acceptable.

## Validation

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Tests use synthetic inputs and small randomly initialized models; they do not
require the pretrained checkpoints. They cover geometry, padding, output
alignment, safe loading from safetensors and `.pt`, latent contracts, and the
NIfTI round trip. Separate checks using the released weights established exact
agreement with the original training implementation, as described in
`VALIDATION.md`.

## Release status

This release provides inference code and three EMA weight bundles. Training
data will not be released. License terms have not yet been specified;
third-party dependencies retain their respective licenses. No patient images,
private data lists, training logs, or credentials are included.
