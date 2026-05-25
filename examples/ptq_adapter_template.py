"""Template adapter for running PTQ on the official Wan2.2 model.

Fill in `load_model`, `run_calibration`, and `proxy_loss` using the exact
pipeline/model APIs from the repository you use for the challenge.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ptq_wan22 import QuantConfig, convert_linear_modules, rank_linear_sensitivity
from ptq_wan22.calibrate import apply_smoothquant, collect_linear_input_stats, tune_clip_ratios
from ptq_wan22.convert import save_quant_state


def load_model(model_dir: str) -> torch.nn.Module:
    # Example shape:
    # from wan import WanI2VPipeline
    # pipe = WanI2VPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to("cuda")
    # return pipe.transformer
    raise NotImplementedError("Wire this to the official Wan2.2 model loader.")


def run_calibration(model: torch.nn.Module) -> None:
    # Run 16-64 representative denoising forwards. Prefer cached latents/text
    # embeddings/images so this calibration is deterministic and fast.
    raise NotImplementedError("Run representative Wan2.2 forward passes here.")


def proxy_loss(model: torch.nn.Module) -> torch.Tensor:
    # Recommended: cache BF16 noise predictions on calibration batches, then
    # return MSE(quant_pred, bf16_pred). Add CLIP/DINO proxy terms if available.
    raise NotImplementedError("Return scalar sensitivity loss here.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--format", default="hif4", choices=["hif4", "mxfp4"])
    parser.add_argument("--out", default="quant_state.pt")
    parser.add_argument("--smooth-alpha", type=float, default=0.5)
    parser.add_argument("--high-precision-blocks", type=int, default=2)
    args = parser.parse_args()

    model = load_model(args.model_dir).eval()
    cfg = QuantConfig(fmt=args.format)
    convert_linear_modules(model, cfg)

    stats = collect_linear_input_stats(model, run_calibration)
    apply_smoothquant(model, stats, alpha=args.smooth_alpha)
    tune_clip_ratios(model, stats, act_quantile=0.999, weight_quantile=0.999)

    ranking = rank_linear_sensitivity(model, proxy_loss)
    keep = [item.name for item in ranking[: args.high_precision_blocks]]
    print("Keep these modules/blocks high precision:")
    for item in ranking[: args.high_precision_blocks]:
        print(f"{item.name}\t{item.score:.6g}")

    save_quant_state(model, args.out)
    Path(args.out).with_suffix(".keep.txt").write_text("\n".join(keep) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
