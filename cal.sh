#!/usr/bin/env bash
set -euo pipefail

cd /mnt/disk2/home/wujianfeng/com/gcc/nvidia

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HIFLOAT4_GPU_DIR="${HIFLOAT4_GPU_DIR:-/mnt/disk2/home/wujianfeng/com/gcc/HiFloat4/hif4_gpu}"
export PYTHONPATH="${HIFLOAT4_GPU_DIR}:${PYTHONPATH:-}"
export PTQ_WAN22_HIF4_BACKEND="${PTQ_WAN22_HIF4_BACKEND:-official}"
export PTQ_WAN22_HIF4_QDIM="${PTQ_WAN22_HIF4_QDIM:--1}"
export PTQ_WAN22_HIF4_FORCE_FP32="${PTQ_WAN22_HIF4_FORCE_FP32:-0}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
ULYSSES_SIZE="${ULYSSES_SIZE:-${NPROC_PER_NODE}}"
QUANTIZE_WEIGHT="${QUANTIZE_WEIGHT:-1}"
QUANTIZE_ACT="${QUANTIZE_ACT:-1}"
CALIB_DATA_ROOT="${CALIB_DATA_ROOT:-HiFloat4/datasets}"
CALIB_SAMPLES="${CALIB_SAMPLES:-2}"
CALIB_STEPS="${CALIB_STEPS:-2}"
DATA_PARALLEL_CALIB="${DATA_PARALLEL_CALIB:-0}"
OFFLOAD_MODEL="${OFFLOAD_MODEL:-1}"
KEEP_FP_REGEX="${KEEP_FP_REGEX:-}"
QUANTIZE_FIRST_LAST_BLOCKS="${QUANTIZE_FIRST_LAST_BLOCKS:-${QUANTIZE_FIRST_LAST_LAYERS:-0}}"
KEEP_FIRST_LAST_LINEAR="${KEEP_FIRST_LAST_LINEAR:-0}"

EXTRA_ARGS=()
if [[ -n "${CALIB_DATA_ROOT}" && "${CALIB_DATA_ROOT}" != "none" ]]; then
  EXTRA_ARGS+=(--calib-data-root "${CALIB_DATA_ROOT}")
fi
if [[ "${QUANTIZE_FIRST_LAST_BLOCKS}" == "1" ]]; then
  EXTRA_ARGS+=(--quantize-first-last-blocks)
fi
if [[ "${KEEP_FIRST_LAST_LINEAR}" == "1" ]]; then
  EXTRA_ARGS+=(--keep-first-last-linear)
fi
if [[ -n "${KEEP_FP_REGEX}" ]]; then
  IFS=',' read -r -a KEEP_PATTERNS <<< "${KEEP_FP_REGEX}"
  for pattern in "${KEEP_PATTERNS[@]}"; do
    if [[ -n "${pattern}" ]]; then
      EXTRA_ARGS+=(--keep-fp-regex "${pattern}")
    fi
  done
fi
if [[ "${QUANTIZE_WEIGHT}" == "0" ]]; then
  EXTRA_ARGS+=(--no-quant-weight)
fi
if [[ "${QUANTIZE_ACT}" == "0" ]]; then
  EXTRA_ARGS+=(--no-quant-act)
fi
if [[ "${DATA_PARALLEL_CALIB}" != "0" ]]; then
  ULYSSES_SIZE=1
  EXTRA_ARGS+=(--data-parallel-calib)
fi
if [[ "${OFFLOAD_MODEL}" == "0" ]]; then
  EXTRA_ARGS+=(--no-offload-model)
fi

echo "[cal] calibration dataset: ${CALIB_DATA_ROOT}"
echo "[cal] calibration samples: ${CALIB_SAMPLES}; denoising steps per sample: ${CALIB_STEPS}"
echo "[cal] data parallel calibration: ${DATA_PARALLEL_CALIB}; ulysses size: ${ULYSSES_SIZE}"
echo "[cal] offload model: ${OFFLOAD_MODEL}"
echo "[cal] keep first/last Transformer blocks BF16: $([[ "${QUANTIZE_FIRST_LAST_BLOCKS}" == "1" ]] && echo no || echo yes)"
echo "[cal] additionally keep first/last Linear modules BF16: $([[ "${KEEP_FIRST_LAST_LINEAR}" == "1" ]] && echo yes || echo no)"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  examples/ptq_wan22_i2v_official.py \
  --wan-repo Wan2.2 \
  --ckpt-dir /mnt/diskhd/Backup/DownloadModel/Wan2.2-I2V-A14B-BF16/ \
  --image /mnt/disk2/home/wujianfeng/com/gcc/OpenS2V-Eval/Images/humanobj/environment/special/21.jpg \
  --format hif4 \
  --frame-num 61 \
  --calib-samples "${CALIB_SAMPLES}" \
  --calib-steps "${CALIB_STEPS}" \
  --ulysses-size "${ULYSSES_SIZE}" \
  --t5-cpu \
  --skip-sensitivity \
  --out outputs/i2v_hif4_quant_state.pt \
  "${EXTRA_ARGS[@]}"
