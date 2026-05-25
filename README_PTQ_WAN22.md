# Wan2.2-I2V-A14B HiFloat4 PTQ

This repository contains a lightweight PTQ workflow for GCC Track 1 on
Wan2.2-I2V-A14B. It wires the official Wan2.2 I2V pipeline to the official
HiFloat4 CUDA simulator and provides scripts for calibration, quantized-weight
export, OpenS2V-Eval generation, and HiFloat4 dataset example generation.

## Quantization Policy

Default policy:

- `pipe.low_noise_model` and `pipe.high_noise_model` are both quantized.
- The first and last Transformer blocks in each expert stay BF16.
- Other `nn.Linear` layers use HiFloat4 W4A4 quant-dequant simulation.
- Non-Linear modules keep the dtype used by Wan2.2.
- Actual GEMM is still BF16 because the official HiFloat4 package exposes a
  quant-dequant simulator, not a packed 4-bit GEMM runtime.

To quantize the first/last blocks too:

```bash
QUANTIZE_FIRST_LAST_BLOCKS=1 bash cal.sh
```

To keep additional modules in BF16, pass comma-separated regexes:

```bash
KEEP_FP_REGEX='low_noise_model.blocks.1,high_noise_model.blocks.1' bash cal.sh
```

Use the same `KEEP_FP_REGEX` and `QUANTIZE_FIRST_LAST_BLOCKS` settings for
calibration, generation, and export.

## Layout

- `ptq_wan22/fp4_formats.py`: MXFP4 fallback and HiFloat4 QDQ entrypoint.
- `ptq_wan22/hifloat4_backend.py`: official `quant_cy` backend wrapper.
- `ptq_wan22/quant_linear.py`: `nn.Linear` replacement with W4A4 QDQ.
- `ptq_wan22/calibrate.py`: activation stats, SmoothQuant, clip tuning, DP stat merge.
- `ptq_wan22/calib_data.py`: HiFloat4/OpenS2V-5M dataset loader and video first-frame cache.
- `ptq_wan22/export.py`: QDQ weight export/load helpers.
- `examples/ptq_wan22_i2v_official.py`: calibration entrypoint.
- `examples/generate_opens2v_eval_wan22.py`: OpenS2V-Eval generation.
- `examples/generate_hifloat4_dataset_examples.py`: sample generation from HiFloat4 dataset records.

## Environment

Build the official HiFloat4 GPU extension:

```bash
cd /mnt/disk2/home/wujianfeng/com/gcc
git clone https://github.com/global-computing-consortium/HiFloat4.git
cd HiFloat4/hif4_gpu
bash build.sh
python hifx4.py
```

The scripts set these defaults:

```bash
HIFLOAT4_GPU_DIR=/mnt/disk2/home/wujianfeng/com/gcc/HiFloat4/hif4_gpu
PTQ_WAN22_HIF4_BACKEND=official
PTQ_WAN22_HIF4_QDIM=-1
PTQ_WAN22_HIF4_FORCE_FP32=0
```

`QDIM=-1` is important for Wan2.2 Linear tensors. Using the small official
example's `dim(0)` on high-dimensional video activations can allocate tens of GB.

## 1. Calibrate Quantization

Default calibration data:

```bash
HiFloat4/datasets/OpenS2V-5M_to_mm.json
```

The dataset JSON uses video `path` and caption `cap`. The loader resolves local
videos under `HiFloat4/datasets/total_part2/...`, extracts the first frame into
`HiFloat4/datasets/.calib_frames/`, and uses the caption as the I2V prompt.

Smoke test:

```bash
cd /mnt/disk2/home/wujianfeng/com/gcc/nvidia
CALIB_SAMPLES=8 CALIB_STEPS=2 bash cal.sh
```

Recommended first full run:

```bash
CALIB_SAMPLES=256 CALIB_STEPS=6 bash cal.sh
```

Stronger run:

```bash
CALIB_SAMPLES=512 CALIB_STEPS=8 bash cal.sh
```

Output:

```bash
outputs/i2v_hif4_quant_state.pt
```

This file is calibration metadata only: SmoothQuant scales and clip ratios. It
does not contain full model weights.

### Data-Parallel Calibration

Pure DP calibration gives each GPU different calibration samples and all-reduces
activation stats at the end. This requires one model replica per GPU.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
DATA_PARALLEL_CALIB=1 \
OFFLOAD_MODEL=1 \
CALIB_SAMPLES=512 \
CALIB_STEPS=4 \
bash cal.sh
```

Set `OFFLOAD_MODEL=0` only if one GPU can hold a full replica:

```bash
OFFLOAD_MODEL=0 DATA_PARALLEL_CALIB=1 NPROC_PER_NODE=8 bash cal.sh
```

## 2. Export Quantized Weights

Official HiFloat4 export writes quant-dequanted BF16 tensors for the low-noise
and high-noise experts separately:

```bash
bash export_wan22_quant_weights.sh
```

Outputs:

```bash
outputs/i2v_hif4_quantized_weights_low_noise_model.pt
outputs/i2v_hif4_quantized_weights_high_noise_model.pt
```

These files are not compressed 4-bit checkpoints. They are BF16 tensors whose
values have already passed through HiFloat4 QDQ. They are useful for avoiding
online weight QDQ during inference; activation QDQ remains enabled by default.

Choose the GPU used for QDQ export:

```bash
CUDA_VISIBLE_DEVICES=4 DEVICE_ID=0 EXPORT_DEVICE=cuda:0 bash export_wan22_quant_weights.sh
```

The repository also has a fallback packed reference format, but it is not the
official HiFloat4 runtime layout:

```bash
MODE=packed PTQ_WAN22_HIF4_BACKEND=fallback bash export_wan22_quant_weights.sh
```

## 3. Generate OpenS2V-Eval Videos

Generate with online quantization state:

```bash
OUT_DIR=/mnt/disk2/home/wujianfeng/com/gcc/OpenS2V-Eval/Generated_Videos/hif4_w4a4 \
bash opens2v_generate.sh
```

Generate with exported QDQ weights:

```bash
LOW_QUANT_WEIGHTS=outputs/i2v_hif4_quantized_weights_low_noise_model.pt \
HIGH_QUANT_WEIGHTS=outputs/i2v_hif4_quantized_weights_high_noise_model.pt \
OUT_DIR=/mnt/disk2/home/wujianfeng/com/gcc/OpenS2V-Eval/Generated_Videos/hif4_exported_w4a4 \
bash opens2v_generate.sh
```

BF16 baseline:

```bash
QUANT_STATE=none \
OUT_DIR=/mnt/disk2/home/wujianfeng/com/gcc/OpenS2V-Eval/Generated_Videos/bf16_baseline \
bash opens2v_generate.sh
```

Small smoke test:

```bash
LIMIT=1 OUT_DIR=/tmp/opens2v_smoke bash opens2v_generate.sh
```

Per-domain override:

```bash
EVAL_JSON=/mnt/disk2/home/wujianfeng/com/gcc/OpenS2V-Eval/Human-Domain_Eval.json \
OUT_DIR=/mnt/disk2/home/wujianfeng/com/gcc/OpenS2V-Eval/Generated_Videos/hif4_human \
bash opens2v_generate.sh
```

OpenS2V-Eval expects generated files to be named by sample id, such as
`singleobj_1.mp4`. The script preserves that naming.

### Data-Parallel OpenS2V Generation

Do not use `DATA_PARALLEL=1 torchrun` for Wan2.2 generation; the official
distributed pipeline returns video only on rank0. Use standalone workers:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUT_DIR=/mnt/disk2/home/wujianfeng/com/gcc/OpenS2V-Eval/Generated_Videos/hif4_dp \
bash opens2v_generate_dp.sh
```

Each worker runs as `WORLD_SIZE=1` on one GPU and processes a different shard.
If single-GPU generation is too slow or OOM, use the normal Ulysses/SP path:

```bash
DATA_PARALLEL=0 NPROC_PER_NODE=4 ULYSSES_SIZE=4 bash opens2v_generate.sh
```

## 4. Generate HiFloat4 Dataset Examples

Generate 64 examples from `HiFloat4/datasets/OpenS2V-5M_to_mm.json`:

```bash
OUT_DIR=/mnt/disk2/home/wujianfeng/com/gcc/OpenS2V-Eval/Generated_Videos/hifloat4_64_examples \
LIMIT=64 \
bash hifloat4_generate_examples.sh
```

Use exported QDQ weights:

```bash
LOW_QUANT_WEIGHTS=outputs/i2v_hif4_quantized_weights_low_noise_model.pt \
HIGH_QUANT_WEIGHTS=outputs/i2v_hif4_quantized_weights_high_noise_model.pt \
OUT_DIR=/mnt/disk2/home/wujianfeng/com/gcc/OpenS2V-Eval/Generated_Videos/hifloat4_64_exported \
LIMIT=64 \
bash hifloat4_generate_examples.sh
```

Standalone worker DP:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUT_DIR=/mnt/disk2/home/wujianfeng/com/gcc/OpenS2V-Eval/Generated_Videos/hifloat4_64_dp \
LIMIT=64 \
bash hifloat4_generate_examples_dp.sh
```

Outputs are named:

```bash
hifloat4_000000.mp4
hifloat4_000001.mp4
...
```

The script writes a manifest for matching generated videos back to source data:


Each row contains `generated`, `index`, `reference_image`, `source_video`,
`source_path`, and `prompt`.

## 5. Evaluation Notes

For Track 1 quality checks, compare BF16 and quantized outputs with the same
JSON, seeds, resolution, frame count, and sample steps. HiF4 target loss is
usually measured as relative VBench score drop versus BF16.

Suggested local sanity sequence:

```bash
QUANT_STATE=none OUT_DIR=/tmp/bf16 LIMIT_PER_TYPE=2 bash opens2v_generate.sh
OUT_DIR=/tmp/hif4 LIMIT_PER_TYPE=2 bash opens2v_generate.sh
```

Then run VBench/VBench-I2V or the official challenge evaluation script on both
directories and compute relative score drop:

```text
drop = (bf16_score - quant_score) / bf16_score
```

## Useful Switches

- `CALIB_SAMPLES=512`: number of calibration samples.
- `CALIB_STEPS=4`: denoising steps per calibration sample.
- `DATA_PARALLEL_CALIB=1`: shard calibration samples across ranks and merge stats.
- `OFFLOAD_MODEL=0`: disable Wan2.2 model offload during calibration/generation.
- `QUANTIZE_ACT=0`: test W4A16 behavior.
- `QUANTIZE_WEIGHT=0`: test A4-only behavior.
- `QUANT_STATE=none`: BF16 baseline generation.
- `LOW_QUANT_WEIGHTS=... HIGH_QUANT_WEIGHTS=...`: use exported QDQ weights.
- `LIMIT=1`: smoke test first sample.
- `LIMIT_PER_TYPE=2`: small balanced OpenS2V subset.
- `SAMPLE_STEPS=...`: override Wan2.2 default generation steps.
- `FRAME_NUM=61`: default evaluation frame count.
