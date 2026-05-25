"""Export Wan2.2 I2V low/high noise QuantLinear weights."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ptq_wan22_i2v_official import (  # noqa: E402
    init_distributed_if_needed,
    load_official_wan_i2v,
)
from ptq_wan22 import (  # noqa: E402
    QuantConfig,
    convert_linear_modules,
    first_last_block_keep_fp_regex,
    first_last_linear_keep_fp_regex,
)
from ptq_wan22.convert import load_quant_state  # noqa: E402
from ptq_wan22.export import (  # noqa: E402
    export_dequantized_linear_weights,
    export_quantized_linear_weights,
)


def _split_path(path: str | Path, expert_name: str) -> Path:
    out = Path(path)
    return out.with_name(f"{out.stem}_{expert_name}{out.suffix}")


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError("--weight-dtype must be bf16, fp16, or fp32")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan-repo", default="Wan2.2")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--quant-state", default="outputs/i2v_hif4_quant_state.pt")
    parser.add_argument("--format", default="hif4", choices=["hif4", "mxfp4"])
    parser.add_argument("--keep-fp-regex", action="append", default=[])
    parser.add_argument("--keep-first-last-linear", action="store_true")
    parser.add_argument("--quantize-first-last-layers", action="store_true")
    parser.add_argument("--quantize-first-last-blocks", action="store_true")
    parser.add_argument("--out", default="outputs/i2v_hif4_quantized_weights.pt")
    parser.add_argument("--mode", default="qdq", choices=["qdq", "packed"])
    parser.add_argument("--weight-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--export-device", default=None, help="CUDA device for official QDQ export, e.g. cuda:0.")
    parser.add_argument("--combined", action="store_true", help="Export both experts into one file.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--ulysses-size", type=int, default=1)
    parser.add_argument("--t5-cpu", action="store_true")
    parser.add_argument("--no-convert-model-dtype", action="store_true")
    parser.add_argument("--no-init-on-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rank, world_size, local_rank = init_distributed_if_needed(args.wan_repo, args.ulysses_size)
    if world_size > 1:
        args.device_id = local_rank

    loaded = load_official_wan_i2v(
        wan_repo=args.wan_repo,
        ckpt_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=rank,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=not args.no_convert_model_dtype,
        init_on_cpu=not args.no_init_on_cpu,
    )

    keep_fp_regex = list(args.keep_fp_regex)
    quantize_first_last_blocks = args.quantize_first_last_layers or args.quantize_first_last_blocks
    if not quantize_first_last_blocks:
        keep_fp_regex.extend(first_last_block_keep_fp_regex(loaded.experts))
    if args.keep_first_last_linear:
        keep_fp_regex.extend(first_last_linear_keep_fp_regex(loaded.experts))
    if rank == 0 and keep_fp_regex:
        print("Keeping BF16 Linear modules matching:")
        for pattern in keep_fp_regex:
            print(f"  {pattern}")

    convert_linear_modules(loaded.experts, QuantConfig(fmt=args.format, keep_fp_regex=keep_fp_regex))
    load_quant_state(loaded.experts, args.quant_state, map_location="cpu")

    if rank == 0:
        exporter = (
            export_dequantized_linear_weights
            if args.mode == "qdq"
            else export_quantized_linear_weights
        )
        common_kwargs = {}
        if args.mode == "qdq":
            common_kwargs["weight_dtype"] = _dtype_from_name(args.weight_dtype)
            common_kwargs["device"] = args.export_device

        if args.combined:
            exporter(loaded.experts, args.out, **common_kwargs)
            print(f"Saved {args.mode} quantized weights: {args.out}")
        else:
            low_out = _split_path(args.out, "low_noise_model")
            high_out = _split_path(args.out, "high_noise_model")
            exporter(loaded.experts.low_noise_model, low_out, **common_kwargs)
            exporter(loaded.experts.high_noise_model, high_out, **common_kwargs)
            print(f"Saved low-noise {args.mode} weights: {low_out}")
            print(f"Saved high-noise {args.mode} weights: {high_out}")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
