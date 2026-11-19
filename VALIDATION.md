# Release validation

Validation used the privately retained training checkpoints. Three release
variants are provided: ViT-VAE 500M, ViT-VAE 250M, and CNN VAE. All use the EMA inference
weights (`payload['ema']`), float32 safetensors export, and the linear [0,1]
pixel head.

| release | source checkpoint | selected state | source step | parameters | tensors | SHA-256 |
|---|---|---|---:|---:|---:|---|
| `vitok-500m-ema` | `ckpt_vitok_500m_kl1e4/ckpts/ckpt_best.pt` | `ema` | 462,000 | 499,260,960 | 328 | `198f1cfe…fb111f` |
| `vitok-250m-ema` | `ckpt_vitok_250m_kl1e4/ckpts/ckpt_latest.pt` | `ema` | 325,000 | 249,197,600 | 168 | `a549ef60…38512` |
| `cnn-246m-ema` | `ckpt_conflux_scratch_perc/ckpts/ckpt_best.pt` | `ema` | 160,000 | 246,463,649 | 236 | `9e4ae3db…b949a3` |

Every exported safetensors tensor was checked for exact equality with the
source checkpoint tensor. No optimizer or discriminator state is stored in the
safetensors files.

## Implementation equivalence

The standalone `vae3d` implementation was compared with the original training
implementations (`vitok_vae` for the ViT models, `conflux.vae` for the CNN) on
synthetic inputs, CPU, float32. Three paths
were checked: the source `.pt` checkpoint, the exported safetensors bundle, and
the high-level `VAEInference` padding/cropping API. All agreed bitwise with the
original implementation in every case.

| model | input shape (D,H,W) | padded shape | latent shape | max abs diff |
|---|---|---|---:|---:|
| 500M | [8, 16, 16] | [8, 16, 16] | [1, 16, 1, 2, 2] | 0.0 |
| 500M | [16, 16, 16] | [16, 16, 16] | [1, 16, 2, 2, 2] | 0.0 |
| 500M | [24, 32, 40] | [24, 32, 40] | [1, 16, 3, 4, 5] | 0.0 |
| 500M | [17, 26, 33] | [24, 32, 40] | [1, 16, 3, 4, 5] | 0.0 |
| 250M | [8, 16, 16] | [8, 16, 16] | [1, 16, 1, 2, 2] | 0.0 |
| 250M | [16, 16, 16] | [16, 16, 16] | [1, 16, 2, 2, 2] | 0.0 |
| 250M | [24, 32, 40] | [24, 32, 40] | [1, 16, 3, 4, 5] | 0.0 |
| 250M | [17, 26, 33] | [24, 32, 40] | [1, 16, 3, 4, 5] | 0.0 |
| CNN | [8, 16, 16] | [8, 16, 16] | [1, 16, 1, 2, 2] | 0.0 |
| CNN | [16, 16, 16] | [16, 16, 16] | [1, 16, 2, 2, 2] | 0.0 |
| CNN | [24, 32, 40] | [24, 32, 40] | [1, 16, 3, 4, 5] | 0.0 |
| CNN | [17, 26, 33] | [24, 32, 40] | [1, 16, 3, 4, 5] | 0.0 |

Machine-readable results, including source and export SHA-256, are retained in
the private validation workspace as `model_source_equivalence.json`; they are
not included in the public code repository.


## Latent normalization statistics

Each model directory includes `latent_stats.json`, computed with
`scripts/compute_latent_stats.py` from the first 1,000 deterministically sampled
training volumes (seed `20261119`, posterior mean, 192x192x144 input grid). These
are explicitly marked `subset_estimate: true`; they are sufficient as an initial
DiT normalization contract but can be recomputed on the full training split.

| model | n entries | latent shape | stats SHA-256 |
|---|---:|---|---|
| ViT-VAE 500M | 1,000 | [16, 24, 24, 18] | `5536541e…d9b608f` |
| ViT-VAE 250M | 1,000 | [16, 24, 24, 18] | `2ee17bea…896fc53` |
| CNN VAE | 1,000 | [16, 24, 24, 18] | `94eac4fc…006309f` |

The contract uses `u = (z - mean) / std_effective`, FP64 population variance,
`eps = 1e-6`, and no latent scale/shift inside the VAE itself.

Eight synthetic tests passed, covering strict safetensors and `.pt` loading,
weight locators, latent geometry, non-divisible input padding, CT boundary
padding, NIfTI physical geometry, preprocessing-config defaults, end-to-end CLI
reconstruction, and refusal to overwrite outputs. Synthetic CT preprocessing
also matched the evaluated preprocessing pipeline exactly. Normalized inputs are
clamped to [0,1] before the model, matching the evaluated model adapter.
