# Wan2.2-I2V-A14B HiFloat4 PTQ

This repository contains the code used for the ICME 2026 Low-Bit-width
Large-Model Quantization Challenge, Track 1. It applies post-training
quantization to Wan2.2-I2V-A14B with the official HiFloat4 simulator.

hugging face: https://huggingface.co/ReopenAI/Wan2.2-I2V-14B-HiF4
modelscope: https://modelscope.cn/models/wjfwjf/Wan2.2-I2V-14B-HiF4

## Method


- Target model: Wan2.2-I2V-A14B.
- Quantized experts: both `low_noise_model` and `high_noise_model`.
- Default precision policy: first and last Transformer blocks stay BF16; other
  `nn.Linear` layers use HiFloat4 W4A4 quant-dequant simulation.
- Runtime math: official HiFloat4 QDQ is used for quantized weights and
  activations; GEMM remains BF16 because the official HiFloat4 package exposes
  a simulator rather than a packed 4-bit GEMM runtime.
- Exported weights: two expert files, one for the low-noise model and one for
  the high-noise model.

More implementation details are in [README_PTQ_WAN22.md](README_PTQ_WAN22.md).

## Repository Layout

- `ptq_wan22/`: quantization modules, calibration, export helpers.
- `examples/ptq_wan22_i2v_official.py`: calibration entrypoint.
- `examples/export_wan22_i2v_quant_weights.py`: low/high expert weight export.
- `examples/generate_opens2v_eval_wan22.py`: OpenS2V-Eval video generation.
- `examples/generate_hifloat4_dataset_examples.py`: HiFloat4 dataset examples.
- `cal.sh`: calibration script.
- `export_wan22_quant_weights.sh`: exported-weight generation script.
- `opens2v_generate.sh`: OpenS2V-Eval generation script.
- `opens2v_generate_dp.sh`: standalone per-GPU generation workers.
- `hifloat4_generate_examples.sh`: generate 64 dataset examples.

## Environment

Create a Python environment compatible with Wan2.2 and install the Python
dependencies:

```bash
pip install -r requirements.txt
```

Build the official HiFloat4 CUDA extension:

```bash
cd /mnt/disk2/home/wujianfeng/com/gcc
git clone https://github.com/global-computing-consortium/HiFloat4.git
cd HiFloat4/hif4_gpu
bash build.sh
python hifx4.py
```

The scripts assume these paths by default:

```bash
WAN_REPO=/mnt/disk2/home/wujianfeng/com/gcc/nvidia/Wan2.2
CKPT_DIR=/mnt/diskhd/Backup/DownloadModel/Wan2.2-I2V-A14B-BF16/
HIFLOAT4_GPU_DIR=/mnt/disk2/home/wujianfeng/com/gcc/HiFloat4/hif4_gpu
OPENS2V_ROOT=/mnt/disk2/home/wujianfeng/com/gcc/OpenS2V-Eval
```

Override `CKPT_DIR`, `HIFLOAT4_GPU_DIR`, `OPENS2V_ROOT`, and output paths with
environment variables if your machine uses a different layout.

## Calibration

Default calibration uses `HiFloat4/datasets/OpenS2V-5M_to_mm.json`. The loader
resolves local videos under `HiFloat4/datasets/total_part2/...`, extracts the
first frame, and uses `cap` as the image-to-video prompt.

Smoke test:

```bash
CALIB_SAMPLES=8 CALIB_STEPS=2 bash cal.sh
```

Recommended reproducible run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
DATA_PARALLEL_CALIB=1 \
OFFLOAD_MODEL=1 \
CALIB_SAMPLES=512 \
CALIB_STEPS=4 \
bash cal.sh
```

Output:

```bash
outputs/i2v_hif4_quant_state.pt
```

This file stores calibration metadata such as SmoothQuant scales and clipping
ratios. It is not a full model checkpoint.

## Export Quantized Weights

Export the two Wan2.2 I2V experts separately:

```bash
bash export_wan22_quant_weights.sh
```

Outputs:

```bash
outputs/i2v_hif4_quantized_weights_low_noise_model.pt
outputs/i2v_hif4_quantized_weights_high_noise_model.pt
```

These are BF16 tensors after HiFloat4 quant-dequant, plus runtime metadata. They
avoid online weight QDQ during inference while keeping activation QDQ enabled.

For a specific physical GPU:

```bash
CUDA_VISIBLE_DEVICES=4 DEVICE_ID=0 EXPORT_DEVICE=cuda:0 bash export_wan22_quant_weights.sh
```

## Inference

Generate OpenS2V-Eval videos with the calibration state:

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

For data-parallel generation, use standalone workers instead of
`DATA_PARALLEL=1 torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUT_DIR=/mnt/disk2/home/wujianfeng/com/gcc/OpenS2V-Eval/Generated_Videos/hif4_dp \
bash opens2v_generate_dp.sh
```

## HiFloat4 Dataset Examples

Generate 64 examples:

```bash
OUT_DIR=/mnt/disk2/home/wujianfeng/com/gcc/OpenS2V-Eval/Generated_Videos/hifloat4_64_examples \
LIMIT=64 \
bash hifloat4_generate_examples.sh
```

Outputs are named `hifloat4_000000.mp4`, `hifloat4_000001.mp4`, and so on. The
script also writes `manifest.jsonl` or `manifest_shard_x_of_n.jsonl` so each
generated video can be matched back to `source_video`, `source_path`, and
`prompt`.

## Evaluation

Use the same OpenS2V-Eval JSON, seed, resolution, frame count, and sample steps
for BF16 and HiFloat4 outputs. Then run the challenge/VBench evaluation on both
directories and compute relative score drop:

```text
drop = (bf16_score - quantized_score) / bf16_score
```
