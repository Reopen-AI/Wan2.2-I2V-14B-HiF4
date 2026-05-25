"""PTQ entrypoint for the official Wan2.2 I2V-A14B repository.

Expected layout:

    E:/nvidia/
      Wan2.2/                         # cloned from https://github.com/Wan-Video/Wan2.2
      Wan2.2-I2V-A14B/                # downloaded checkpoint directory
      ptq_wan22/
      examples/ptq_wan22_i2v_official.py

This script follows the official generate.py loading path:

    cfg = WAN_CONFIGS["i2v-A14B"]
    pipe = wan.WanI2V(config=cfg, checkpoint_dir=..., ...)

Then it converts both MoE DiT experts:

    pipe.low_noise_model
    pipe.high_noise_model
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ptq_wan22 import (
    QuantConfig,
    convert_linear_modules,
    first_last_block_keep_fp_regex,
    first_last_linear_keep_fp_regex,
    rank_linear_sensitivity,
)
from ptq_wan22.calib_data import CalibrationItem, load_calibration_items
from ptq_wan22.calibrate import (
    apply_smoothquant,
    collect_linear_input_stats,
    merge_distributed_act_stats,
    tune_clip_ratios,
)
from ptq_wan22.convert import save_quant_state


def _progress(iterable, *, desc: str, total: int | None = None, enabled: bool = True):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm(
            iterable,
            desc=desc,
            total=total,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                "[elapsed {elapsed} < remaining {remaining}, {rate_fmt}]"
            ),
        )
    except Exception:
        def _logged():
            for idx, item in enumerate(iterable, start=1):
                suffix = f"/{total}" if total is not None else ""
                print(f"[{desc}] {idx}{suffix}")
                yield item

        return _logged()


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


DEFAULT_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
    "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
    "Blurred beach scenery forms the background featuring crystal-clear waters, distant "
    "green hills, and a blue sky dotted with white clouds."
)


class WanI2VExperts(nn.Module):
    """Module wrapper so PTQ utilities can traverse both official DiT experts."""

    def __init__(self, pipe: Any) -> None:
        super().__init__()
        self.low_noise_model = pipe.low_noise_model
        self.high_noise_model = pipe.high_noise_model


@dataclass
class LoadedWanI2V:
    pipe: Any
    experts: WanI2VExperts
    cfg: Any


def add_wan_repo_to_path(wan_repo: str | Path) -> Path:
    repo = Path(wan_repo).resolve()
    if not (repo / "wan").exists():
        raise FileNotFoundError(f"Cannot find official Wan package under: {repo}")
    sys.path.insert(0, str(repo))
    return repo


def load_official_wan_i2v(
    wan_repo: str | Path,
    ckpt_dir: str | Path,
    device_id: int = 0,
    rank: int = 0,
    t5_fsdp: bool = False,
    dit_fsdp: bool = False,
    use_sp: bool = False,
    t5_cpu: bool = False,
    convert_model_dtype: bool = True,
    init_on_cpu: bool = True,
) -> LoadedWanI2V:
    """Load Wan2.2-I2V-A14B exactly through the official WanI2V path."""

    add_wan_repo_to_path(wan_repo)

    import wan  # type: ignore
    from wan.configs import WAN_CONFIGS  # type: ignore

    cfg = WAN_CONFIGS["i2v-A14B"]
    pipe = wan.WanI2V(
        config=cfg,
        checkpoint_dir=str(ckpt_dir),
        device_id=device_id,
        rank=rank,
        t5_fsdp=t5_fsdp,
        dit_fsdp=dit_fsdp,
        use_sp=use_sp,
        t5_cpu=t5_cpu,
        init_on_cpu=init_on_cpu,
        convert_model_dtype=convert_model_dtype,
    )
    experts = WanI2VExperts(pipe).eval().requires_grad_(False)
    return LoadedWanI2V(pipe=pipe, experts=experts, cfg=cfg)


def init_distributed_if_needed(wan_repo: str | Path, ulysses_size: int) -> tuple[int, int, int]:
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if world_size <= 1:
        if ulysses_size > 1:
            raise ValueError("--ulysses-size > 1 requires torchrun with multiple processes.")
        return rank, world_size, local_rank

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )

    if ulysses_size > 1:
        if ulysses_size != world_size:
            raise ValueError("--ulysses-size must equal WORLD_SIZE for Wan2.2 sequence parallel.")
        add_wan_repo_to_path(wan_repo)
        from wan.distributed.util import init_distributed_group  # type: ignore

        init_distributed_group()

    return rank, world_size, local_rank


def _parse_size(size: str) -> tuple[int, int]:
    width, height = size.lower().split("*")
    return int(width), int(height)


def _run_generate_for_hooks(
    pipe: Any,
    prompt: str,
    image_path: str | Path,
    max_area: int,
    frame_num: int,
    sampling_steps: int,
    seed: int,
    offload_model: bool,
    shift: float,
    guide_scale: tuple[float, float],
) -> None:
    image = Image.open(image_path).convert("RGB")
    _ = pipe.generate(
        prompt,
        image,
        max_area=max_area,
        frame_num=frame_num,
        shift=shift,
        sample_solver="unipc",
        sampling_steps=sampling_steps,
        guide_scale=guide_scale,
        seed=seed,
        offload_model=offload_model,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_calibration_runner(args: argparse.Namespace, pipe: Any, cfg: Any):
    max_area = args.max_area if args.max_area is not None else math.prod(_parse_size(args.size))
    guide_scale = tuple(cfg.sample_guide_scale)
    show_progress = int(os.getenv("RANK", "0")) == 0
    dp_rank = int(os.getenv("RANK", "0")) if args.data_parallel_calib else 0
    dp_world_size = int(os.getenv("WORLD_SIZE", "1")) if args.data_parallel_calib else 1
    if args.data_parallel_calib and dp_rank != 0 and dist.is_available() and dist.is_initialized():
        dist.barrier()
    if args.calib_data_root is not None:
        calib_items = load_calibration_items(
            args.calib_data_root,
            limit=None if args.data_parallel_calib else args.calib_samples,
            show_progress=show_progress,
        )
    else:
        calib_items = [CalibrationItem(image=Path(args.image), prompt=args.prompt)]
    if args.data_parallel_calib and dp_rank == 0 and dist.is_available() and dist.is_initialized():
        dist.barrier()
    if args.data_parallel_calib:
        selected_items = calib_items[: args.calib_samples]
        calib_items = selected_items[dp_rank::dp_world_size]
        local_samples = len(calib_items)
        if not calib_items:
            raise ValueError(
                f"Rank {dp_rank} has no calibration samples. Reduce WORLD_SIZE or increase CALIB_SAMPLES."
            )
        if show_progress:
            print(
                f"Data-parallel calibration rank {dp_rank}/{dp_world_size}: "
                f"{local_samples} local samples from {len(selected_items)} total samples"
            )
    else:
        local_samples = args.calib_samples
    if show_progress:
        print(f"Loaded {len(calib_items)} calibration items")

    def run_calibration(_: nn.Module) -> None:
        indices = range(local_samples)
        sample_durations: list[float] = []
        for idx in _progress(
            indices,
            desc="calibration generates",
            total=local_samples,
            enabled=show_progress,
        ):
            if show_progress and sample_durations:
                avg_seconds = sum(sample_durations) / len(sample_durations)
                remaining = avg_seconds * (local_samples - idx)
                eta = datetime.now() + timedelta(seconds=remaining)
                print(
                    "[cal] "
                    f"sample {idx + 1}/{local_samples}; "
                    f"avg {avg_seconds:.1f}s/sample; "
                    f"remaining {_format_duration(remaining)}; "
                    f"ETA {eta.strftime('%Y-%m-%d %H:%M:%S')}",
                    flush=True,
                )
            elif show_progress:
                print(
                    f"[cal] sample {idx + 1}/{local_samples}; estimating ETA after first sample",
                    flush=True,
                )

            item = calib_items[idx % len(calib_items)]
            start = time.perf_counter()
            _run_generate_for_hooks(
                pipe=pipe,
                prompt=item.prompt,
                image_path=item.image,
                max_area=max_area,
                frame_num=args.frame_num,
                sampling_steps=args.calib_steps,
                seed=args.seed + dp_rank * local_samples + idx,
                offload_model=args.offload_model,
                shift=args.sample_shift,
                guide_scale=guide_scale,
            )
            sample_durations.append(time.perf_counter() - start)

    return run_calibration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan-repo", default="Wan2.2", help="Path to official Wan2.2 repo.")
    parser.add_argument("--ckpt-dir", required=True, help="Path to Wan2.2-I2V-A14B checkpoint dir.")
    parser.add_argument("--image", default="Wan2.2/examples/i2v_input.JPG")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--calib-data-root", default=None)
    parser.add_argument("--size", default="1280*720")
    parser.add_argument("--max-area", type=int, default=None)
    parser.add_argument("--frame-num", type=int, default=9, help="Use 4n+1. Challenge eval may use 61.")
    parser.add_argument("--calib-samples", type=int, default=2)
    parser.add_argument("--calib-steps", type=int, default=2)
    parser.add_argument("--data-parallel-calib", action="store_true")
    parser.add_argument("--sample-shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--ulysses-size", type=int, default=1)
    parser.add_argument("--dit-fsdp", action="store_true")
    parser.add_argument("--t5-fsdp", action="store_true")
    parser.add_argument("--format", default="hif4", choices=["hif4", "mxfp4"])
    parser.add_argument("--keep-fp-regex", action="append", default=[])
    parser.add_argument("--keep-first-last-linear", action="store_true")
    parser.add_argument("--quantize-first-last-layers", action="store_true")
    parser.add_argument("--quantize-first-last-blocks", action="store_true")
    parser.add_argument("--no-quant-weight", action="store_true")
    parser.add_argument("--no-quant-act", action="store_true")
    parser.add_argument("--smooth-alpha", type=float, default=0.5)
    parser.add_argument("--high-precision-blocks", type=int, default=2)
    parser.add_argument("--out", default="outputs/i2v_hif4_quant_state.pt")
    parser.add_argument("--t5-cpu", action="store_true")
    parser.add_argument("--no-convert-model-dtype", action="store_true")
    parser.add_argument("--no-init-on-cpu", action="store_true")
    parser.add_argument("--no-offload-model", action="store_true")
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="Only calibrate and save quant state. Sensitivity requires a custom proxy loss.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rank, world_size, local_rank = init_distributed_if_needed(args.wan_repo, args.ulysses_size)
    if rank != 0:
        os.environ.setdefault("TQDM_DISABLE", "1")
    if world_size > 1:
        args.device_id = local_rank
    elif args.dit_fsdp or args.t5_fsdp:
        raise ValueError("--dit-fsdp/--t5-fsdp require torchrun with multiple processes.")
    if args.dit_fsdp:
        raise ValueError(
            "This PTQ script replaces nn.Linear modules after WanI2V construction, which is "
            "not compatible with DiT FSDP wrapping. Use --ulysses-size without --dit-fsdp, "
            "or integrate convert_linear_modules before Wan2.2 calls shard_model()."
        )

    args.offload_model = not args.no_offload_model
    loaded = load_official_wan_i2v(
        wan_repo=args.wan_repo,
        ckpt_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
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

    qcfg = QuantConfig(
        fmt=args.format,
        keep_fp_regex=keep_fp_regex,
        quantize_weight=not args.no_quant_weight,
        quantize_act=not args.no_quant_act,
    )
    convert_linear_modules(loaded.experts, qcfg)

    run_calibration = build_calibration_runner(args, loaded.pipe, loaded.cfg)
    stats = collect_linear_input_stats(loaded.experts, run_calibration)
    if args.data_parallel_calib:
        if rank == 0:
            print("Merging activation stats across data-parallel ranks...")
        stats = merge_distributed_act_stats(stats)
    apply_smoothquant(loaded.experts, stats, alpha=args.smooth_alpha)
    tune_clip_ratios(loaded.experts, stats)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        save_quant_state(loaded.experts, str(out))

    if args.skip_sensitivity:
        if rank == 0:
            print(f"Saved quant state: {out}")
            print("Skipped sensitivity ranking. Add a cached BF16-vs-quant proxy loss for final block selection.")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    raise NotImplementedError(
        "Official loading and calibration are wired. For sensitivity ranking, cache BF16 DiT "
        "predictions from representative denoising batches and pass an MSE proxy_loss into "
        "rank_linear_sensitivity(...)."
    )


if __name__ == "__main__":
    main()
