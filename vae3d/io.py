"""NIfTI preprocessing with ANTs geometry retained on the output 1 mm grid."""
from dataclasses import dataclass, fields
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class PreprocessConfig:
    """Evaluated inference preprocessing hyperparameters.

    The defaults are the frozen paper contract.  ``configs/preprocess.json``
    contains the same values for machine-readable distribution with the weights.
    """

    orientation: str = "RAS"
    target_spacing_mm: tuple = (1.0, 1.0, 1.0)
    ants_interpolation_code: int = 1
    use_voxels: bool = False
    array_order: str = "DHW"
    intensity_clip_percentiles: tuple = (0.5, 99.5)
    normalization_eps: float = 1e-8
    image_normalization: str = "percentile_clip_to_0_1"
    input_range: tuple = (0.0, 1.0)
    padding_multiple: tuple = (8, 8, 8)
    padding_value: float = 0.0
    padding_mode: str = "symmetric_zero"
    ct_padding_repair: bool = True
    ct_sentinel_hu: float = -3024.0
    ct_air_hu: float = -1000.0
    latent_normalization: str = "none"
    output_scale: str = "source"

    def __post_init__(self):
        object.__setattr__(self, "target_spacing_mm", tuple(float(x) for x in self.target_spacing_mm))
        object.__setattr__(self, "intensity_clip_percentiles",
                           tuple(float(x) for x in self.intensity_clip_percentiles))
        object.__setattr__(self, "input_range", tuple(float(x) for x in self.input_range))
        object.__setattr__(self, "padding_multiple", tuple(int(x) for x in self.padding_multiple))
        if self.orientation != "RAS":
            raise ValueError("Only RAS orientation is supported by the evaluated pipeline")
        if self.array_order != "DHW":
            raise ValueError("Only DHW array order is supported")
        low, high = self.intensity_clip_percentiles
        if not 0 <= low < high <= 100:
            raise ValueError("intensity_clip_percentiles must satisfy 0 <= low < high <= 100")
        if len(self.target_spacing_mm) != 3 or any(x <= 0 for x in self.target_spacing_mm):
            raise ValueError("target_spacing_mm must contain three positive values")
        if len(self.padding_multiple) != 3 or any(x <= 0 for x in self.padding_multiple):
            raise ValueError("padding_multiple must contain three positive integers")
        if self.input_range != (0.0, 1.0):
            raise ValueError("Released models expect input_range [0,1]")
        if self.image_normalization != "percentile_clip_to_0_1":
            raise ValueError("Only percentile_clip_to_0_1 image normalization is supported")
        if self.padding_mode != "symmetric_zero":
            raise ValueError("Only symmetric_zero padding is supported")
        if self.latent_normalization != "none":
            raise ValueError("The released VAE applies no latent scale or shift")
        if self.output_scale not in ("source", "normalized"):
            raise ValueError("output_scale must be source or normalized")


def load_preprocess_config(path):
    raw = json.loads(Path(path).read_text())
    allowed = {f.name for f in fields(PreprocessConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown preprocessing fields: {sorted(unknown)}")
    return PreprocessConfig(**raw)


@dataclass
class PreparedVolume:
    array01: np.ndarray
    reference: object
    low: float
    high: float
    padding_fraction: float
    eps: float = 1e-8


def repair_ct_padding(values, sentinel=-3024.0, air=-1000.0):
    """Replace only exact sentinel voxels connected to any array boundary."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 3 or not np.isfinite(arr).all():
        raise ValueError("Expected a finite 3D HU array")
    candidate = arr == sentinel
    seeds = np.zeros(arr.shape, dtype=bool)
    for axis in range(3):
        for edge in (0, -1):
            sl = [slice(None)] * 3
            sl[axis] = edge
            seeds[tuple(sl)] = candidate[tuple(sl)]
    padding = ndimage.binary_propagation(seeds, mask=candidate)
    result = arr.copy()
    result[padding] = air
    return result, float(padding.mean())


def prepare_nifti(path, modality="mri", ct_padding=None, config=None):
    """Replicate the evaluated 1 mm / percentile / [0,1] preprocessing.

    ``ct_padding`` defaults to ``config.ct_padding_repair``.  Pass ``False`` to
    disable the CT boundary-padding repair, or ``True`` to force it.
    """
    import ants
    if modality not in ("mri", "ct"):
        raise ValueError("modality must be mri or ct")
    cfg = config or PreprocessConfig()
    if ct_padding is None:
        ct_padding = cfg.ct_padding_repair
    p_low, p_high = cfg.intensity_clip_percentiles
    image = ants.image_read(str(path))
    if image.dimension != 3:
        raise ValueError("Expected a single 3D volume; split MRI sequences before inference")
    values = image.numpy().astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Input contains nonfinite intensities")
    fraction = 0.0
    if modality == "ct" and ct_padding:
        values, fraction = repair_ct_padding(values, cfg.ct_sentinel_hu, cfg.ct_air_hu)
    image = ants.from_numpy(values, origin=image.origin, spacing=image.spacing, direction=image.direction)
    image = ants.reorient_image2(image, cfg.orientation)
    # ANTs code 1 is nearest-neighbor, retained to match evaluated preprocessing.
    image = ants.resample_image(image, cfg.target_spacing_mm,
                                use_voxels=cfg.use_voxels,
                                interp_type=cfg.ants_interpolation_code)
    arr = np.ascontiguousarray(image.numpy().transpose(2, 1, 0), dtype=np.float32)
    low, high = np.percentile(arr, [p_low, p_high])
    if not high > low:
        raise ValueError("Degenerate percentile range")
    np.clip(arr, low, high, out=arr)
    arr -= low
    arr /= high - low + cfg.normalization_eps
    # Float32 percentile arithmetic can overshoot an endpoint by one ULP.
    # The evaluated model adapter also clamps to the [0,1] input convention.
    np.clip(arr, 0.0, 1.0, out=arr)
    return PreparedVolume(arr, image, float(low), float(high), fraction, cfg.normalization_eps)


def save_reconstruction(array01, prepared, output, scale="source"):
    import ants
    arr = np.asarray(array01, dtype=np.float32)
    if arr.shape != prepared.array01.shape or not np.isfinite(arr).all():
        raise ValueError("Reconstruction must be finite and match the prepared grid")
    if scale not in ("source", "normalized"):
        raise ValueError("scale must be source or normalized")
    if scale == "source":
        arr = arr * (prepared.high - prepared.low) + prepared.low
    ref = prepared.reference
    image = ants.from_numpy(np.ascontiguousarray(arr.transpose(2, 1, 0)),
                            origin=ref.origin, spacing=ref.spacing, direction=ref.direction)
    ants.image_write(image, str(output))
