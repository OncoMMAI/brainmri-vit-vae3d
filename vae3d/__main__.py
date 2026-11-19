import argparse
from pathlib import Path
import torch
from .inference import VAEInference, load_model
from .io import load_preprocess_config, prepare_nifti, save_reconstruction


def main():
    p = argparse.ArgumentParser(description="Reconstruct one 3D NIfTI with a released 500M/250M ViT-VAE or CNN VAE")
    p.add_argument("--weights", required=True, type=Path,
                   help=".safetensors, a self-describing .pt checkpoint, or a .path locator")
    p.add_argument("--config", type=Path, default=None,
                   help="JSON architecture config; required for .safetensors, optional for .pt")
    p.add_argument("--variant", choices=["ema", "model"], default="ema",
                   help="checkpoint state to load when --weights is a .pt file")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--modality", choices=["mri", "ct"], required=True)
    p.add_argument("--preprocess-config", type=Path, default=None,
                   help="JSON preprocessing hyperparameters (configs/preprocess.json)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--output-scale", choices=["source", "normalized"], default=None,
                   help="default: preprocess-config output_scale, or source")
    p.add_argument("--no-ct-padding-repair", action="store_true")
    args = p.parse_args()
    if args.output.exists() or args.input.resolve() == args.output.resolve():
        p.error("Output must be a new file, distinct from input")
    if not str(args.output).endswith((".nii", ".nii.gz")):
        p.error("Output must end in .nii or .nii.gz")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    pp = load_preprocess_config(args.preprocess_config) if args.preprocess_config else None
    prepared = prepare_nifti(args.input, args.modality,
                             False if args.no_ct_padding_repair else None, config=pp)
    model = load_model(args.weights, args.config, args.device, args.variant)
    if pp is not None and tuple(pp.padding_multiple) != tuple(model.config.patch_size):
        p.error("Preprocess padding_multiple must match the model patch_size")
    runner = VAEInference(model, args.precision)
    try:
        recon = runner.reconstruct(prepared.array01)
    except torch.cuda.OutOfMemoryError:
        raise SystemExit("Full-volume inference exceeded GPU memory. Use a larger GPU; input was not cropped or tiled.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_scale = args.output_scale or (pp.output_scale if pp is not None else "source")
    save_reconstruction(recon, prepared, args.output, output_scale)
    print(f"Reconstructed {prepared.array01.shape} on the 1 mm grid; output scale={output_scale}")


if __name__ == "__main__":
    main()
