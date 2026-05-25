"""Generate Wan2.2 I2V videos in OpenS2V-Eval filename format.

OpenS2V-Eval expects generated videos to be named by sample id:

    Generated_Videos/
      singleobj_1.mp4
      humanobj_15.mp4
      ...

This script reads one OpenS2V-Eval json file and writes exactly that layout.
For samples with multiple reference images, WanI2V uses the first image by
default because the official I2V entrypoint accepts a single reference image.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
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
)
from ptq_wan22.convert import load_quant_state
from ptq_wan22.export import load_dequantized_linear_weights


class WanI2VExperts(nn.Module):
    def __init__(self, pipe: Any) -> None:
        super().__init__()
        self.low_noise_model = pipe.low_noise_model
        self.high_noise_model = pipe.high_noise_model


def add_repo_to_path(path: str | Path) -> Path:
    repo = Path(path).resolve()
    if not (repo / "wan").exists():
        raise FileNotFoundError(f"Cannot find Wan2.2 repo: {repo}")
    sys.path.insert(0, str(repo))
    return repo


def init_dist(wan_repo: str | Path, ulysses_size: int) -> tuple[int, int, int]:
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size <= 1:
        if ulysses_size > 1:
            raise ValueError("--ulysses-size > 1 requires torchrun.")
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
            raise ValueError("--ulysses-size must equal WORLD_SIZE.")
        add_repo_to_path(wan_repo)
        from wan.distributed.util import init_distributed_group  # type: ignore

        init_distributed_group()
    return rank, world_size, local_rank


def _video_type(video_id: str) -> str:
    prefix, sep, suffix = video_id.rpartition("_")
    if sep and suffix.isdigit():
        return prefix
    return video_id


def load_eval_items(
    eval_json: str | Path,
    limit: int | None = None,
    limit_per_type: int | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    data = json.loads(Path(eval_json).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("OpenS2V eval json should be a dict keyed by video id.")
    items = list(data.items())
    if limit_per_type is not None:
        counts: dict[str, int] = {}
        kept: list[tuple[str, dict[str, Any]]] = []
        for video_id, item in items:
            group = _video_type(video_id)
            count = counts.get(group, 0)
            if count >= limit_per_type:
                continue
            kept.append((video_id, item))
            counts[group] = count + 1
        items = kept
    if limit is not None:
        items = items[:limit]
    return items


def resolve_image(image_root: str | Path, img_paths: list[str]) -> Path:
    if not img_paths:
        raise ValueError("Sample has empty img_paths.")
    path = Path(img_paths[0])
    if not path.is_absolute():
        path = Path(image_root) / path
    if not path.exists():
        raise FileNotFoundError(f"Cannot find reference image: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan-repo", default="Wan2.2")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--image-root", required=True, help="OpenS2V-Eval root containing Images/.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--format", default="hif4", choices=["hif4", "mxfp4"])
    parser.add_argument("--quant-state", default=None)
    parser.add_argument("--low-quant-weights", default=None)
    parser.add_argument("--high-quant-weights", default=None)
    parser.add_argument("--keep-fp-regex", action="append", default=[])
    parser.add_argument("--keep-first-last-linear", action="store_true")
    parser.add_argument("--quantize-first-last-layers", action="store_true")
    parser.add_argument("--quantize-first-last-blocks", action="store_true")
    parser.add_argument("--no-quant-weight", action="store_true")
    parser.add_argument("--no-quant-act", action="store_true")
    parser.add_argument("--size", default="1280*720")
    parser.add_argument("--frame-num", type=int, default=61)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--sample-shift", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-type", type=int, default=None)
    parser.add_argument("--eval-shard-index", type=int, default=None)
    parser.add_argument("--eval-num-shards", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--ulysses-size", type=int, default=1)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--dit-fsdp", action="store_true")
    parser.add_argument("--t5-fsdp", action="store_true")
    parser.add_argument("--t5-cpu", action="store_true")
    parser.add_argument("--convert-model-dtype", action="store_true")
    parser.add_argument("--offload-model", action="store_true")
    args = parser.parse_args()

    if args.data_parallel and args.ulysses_size != 1:
        raise ValueError("--data-parallel requires --ulysses-size 1.")

    rank, world_size, local_rank = init_dist(args.wan_repo, args.ulysses_size)
    device_id = local_rank if world_size > 1 else 0
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    add_repo_to_path(args.wan_repo)
    import wan  # type: ignore
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS  # type: ignore
    from wan.utils.utils import save_video  # type: ignore

    use_exported_weights = args.low_quant_weights is not None or args.high_quant_weights is not None
    use_quantized_path = args.quant_state is not None or use_exported_weights
    if use_quantized_path and args.dit_fsdp:
        raise ValueError(
            "--dit-fsdp is only supported for the unquantized BF16 path. "
            "The PTQ path replaces Linear modules after WanI2V construction, "
            "which is not compatible with already-wrapped DiT FSDP modules."
        )

    cfg = WAN_CONFIGS["i2v-A14B"]
    pipe = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device_id,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    if use_exported_weights and (args.low_quant_weights is None or args.high_quant_weights is None):
        raise ValueError("--low-quant-weights and --high-quant-weights must be provided together.")

    if args.quant_state is not None or use_exported_weights:
        experts = WanI2VExperts(pipe)
        keep_fp_regex = list(args.keep_fp_regex)
        quantize_first_last_blocks = args.quantize_first_last_layers or args.quantize_first_last_blocks
        if not quantize_first_last_blocks:
            keep_fp_regex.extend(first_last_block_keep_fp_regex(experts))
        if args.keep_first_last_linear:
            keep_fp_regex.extend(first_last_linear_keep_fp_regex(experts))
        if rank == 0 and keep_fp_regex:
            print("Keeping BF16 Linear modules matching:")
            for pattern in keep_fp_regex:
                print(f"  {pattern}")
        convert_linear_modules(
            experts,
            QuantConfig(
                fmt=args.format,
                keep_fp_regex=keep_fp_regex,
                quantize_weight=(not args.no_quant_weight) and (not use_exported_weights),
                quantize_act=not args.no_quant_act,
            ),
        )
        if use_exported_weights:
            load_dequantized_linear_weights(experts.low_noise_model, args.low_quant_weights)
            load_dequantized_linear_weights(experts.high_noise_model, args.high_quant_weights)
        else:
            load_quant_state(experts, args.quant_state, map_location="cpu")

    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    all_items = load_eval_items(args.eval_json, args.limit, args.limit_per_type)
    indexed_items = list(enumerate(all_items))
    full_total_items = len(indexed_items)
    if args.eval_shard_index is not None or args.eval_num_shards is not None:
        if args.eval_shard_index is None or args.eval_num_shards is None:
            raise ValueError("--eval-shard-index and --eval-num-shards must be provided together.")
        if not 0 <= args.eval_shard_index < args.eval_num_shards:
            raise ValueError("--eval-shard-index must be in [0, --eval-num-shards).")
        indexed_items = indexed_items[args.eval_shard_index :: args.eval_num_shards]
    total_items = len(indexed_items)
    if args.data_parallel and world_size > 1:
        indexed_items = indexed_items[rank::world_size]
        print(f"[rank{rank}] data-parallel shard: {len(indexed_items)}/{total_items} items")
    max_area = MAX_AREA_CONFIGS[args.size]
    steps = args.sample_steps if args.sample_steps is not None else cfg.sample_steps
    shift = args.sample_shift if args.sample_shift is not None else cfg.sample_shift

    for local_index, (global_index, (video_id, item)) in enumerate(indexed_items):
        out_file = out_dir / f"{video_id}.mp4"
        if args.skip_existing and out_file.exists():
            print(f"[skip] {out_file}")
        skip = [args.skip_existing and out_file.exists()]
        if dist.is_available() and dist.is_initialized() and not args.data_parallel:
            dist.broadcast_object_list(skip, src=0)
        if skip[0]:
            continue

        prompt = item["prompt"]
        image_path = resolve_image(args.image_root, item["img_paths"])
        image = Image.open(image_path).convert("RGB")

        print(
            f"[rank{rank}] [{local_index + 1}/{len(indexed_items)} local, "
            f"{global_index + 1}/{full_total_items} global] {video_id}: {image_path}"
        )

        video = pipe.generate(
            prompt,
            image,
            max_area=max_area,
            frame_num=args.frame_num,
            shift=shift,
            sample_solver="unipc",
            sampling_steps=steps,
            guide_scale=tuple(cfg.sample_guide_scale),
            seed=args.seed + global_index,
            offload_model=args.offload_model,
        )

        if video is None:
            print(
                f"[rank{rank}] pipe.generate returned None for {video_id}; "
                "this Wan2.2 pipeline only returns videos on rank0 under torch.distributed. "
                "Use standalone per-GPU workers for data-parallel generation.",
                flush=True,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if rank == 0 or args.data_parallel:
            save_video(
                tensor=video[None],
                save_file=str(out_file),
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
        del video
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
