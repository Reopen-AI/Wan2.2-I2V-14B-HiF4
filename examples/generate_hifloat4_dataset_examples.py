"""Generate Wan2.2 I2V examples from HiFloat4/OpenS2V-5M dataset records."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate_opens2v_eval_wan22 import WanI2VExperts, add_repo_to_path, init_dist  # noqa: E402
from ptq_wan22 import (  # noqa: E402
    QuantConfig,
    convert_linear_modules,
    first_last_block_keep_fp_regex,
    first_last_linear_keep_fp_regex,
)
from ptq_wan22.calib_data import load_calibration_items  # noqa: E402
from ptq_wan22.convert import load_quant_state  # noqa: E402
from ptq_wan22.export import load_dequantized_linear_weights  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan-repo", default="Wan2.2")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--dataset-root", default="HiFloat4/datasets")
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
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--eval-shard-index", type=int, default=None)
    parser.add_argument("--eval-num-shards", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--ulysses-size", type=int, default=1)
    parser.add_argument("--t5-cpu", action="store_true")
    parser.add_argument("--convert-model-dtype", action="store_true")
    parser.add_argument("--offload-model", action="store_true")
    args = parser.parse_args()

    rank, world_size, local_rank = init_dist(args.wan_repo, args.ulysses_size)
    device_id = local_rank if world_size > 1 else 0
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    add_repo_to_path(args.wan_repo)
    import wan  # type: ignore
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS  # type: ignore
    from wan.utils.utils import save_video  # type: ignore

    cfg = WAN_CONFIGS["i2v-A14B"]
    pipe = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device_id,
        rank=rank,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    use_exported_weights = args.low_quant_weights is not None or args.high_quant_weights is not None
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
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    load_limit = None if args.eval_num_shards is not None else args.limit
    items = load_calibration_items(args.dataset_root, limit=load_limit, show_progress=(rank == 0))
    indexed_items = list(enumerate(items[: args.limit]))
    full_total = len(indexed_items)
    if args.eval_shard_index is not None or args.eval_num_shards is not None:
        if args.eval_shard_index is None or args.eval_num_shards is None:
            raise ValueError("--eval-shard-index and --eval-num-shards must be provided together.")
        indexed_items = indexed_items[args.eval_shard_index :: args.eval_num_shards]

    max_area = MAX_AREA_CONFIGS[args.size]
    steps = args.sample_steps if args.sample_steps is not None else cfg.sample_steps
    shift = args.sample_shift if args.sample_shift is not None else cfg.sample_shift
    if args.eval_shard_index is None:
        manifest_path = out_dir / "manifest.jsonl"
    else:
        manifest_path = out_dir / f"manifest_shard_{args.eval_shard_index}_of_{args.eval_num_shards}.jsonl"
    if rank == 0:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("", encoding="utf-8")

    for local_index, (global_index, item) in enumerate(indexed_items):
        out_file = out_dir / f"hifloat4_{global_index:06d}.mp4"
        if args.skip_existing and out_file.exists():
            print(f"[rank{rank}] [skip] {out_file}")
            continue

        image = Image.open(item.image).convert("RGB")
        print(
            f"[rank{rank}] [{local_index + 1}/{len(indexed_items)} local, "
            f"{global_index + 1}/{full_total} global] {out_file.name}: {item.image}"
        )
        video = pipe.generate(
            item.prompt,
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
            print(f"[rank{rank}] pipe.generate returned None for {out_file.name}; skipping")
            continue
        save_video(
            tensor=video[None],
            save_file=str(out_file),
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "generated": str(out_file),
                        "generated_name": out_file.name,
                        "index": global_index,
                        "reference_image": str(item.image),
                        "source_video": None if item.source_video is None else str(item.source_video),
                        "source_path": item.source_path,
                        "prompt": item.prompt,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        del video
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
