#!/usr/bin/env bash
set -euo pipefail

cd /mnt/disk2/home/wujianfeng/com/gcc/nvidia

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HIFLOAT4_GPU_DIR="${HIFLOAT4_GPU_DIR:-/mnt/disk2/home/wujianfeng/com/gcc/HiFloat4/hif4_gpu}"
export PYTHONPATH="${HIFLOAT4_GPU_DIR}:${PYTHONPATH:-}"
export PTQ_WAN22_HIF4_BACKEND="${PTQ_WAN22_HIF4_BACKEND:-official}"
export PTQ_WAN22_HIF4_QDIM="${PTQ_WAN22_HIF4_QDIM:--1}"
export PTQ_WAN22_HIF4_FORCE_FP32="${PTQ_WAN22_HIF4_FORCE_FP32:-0}"

CKPT_DIR="${CKPT_DIR:-/mnt/diskhd/Backup/DownloadModel/Wan2.2-I2V-A14B-BF16/}"
QUANT_STATE="${QUANT_STATE:-outputs/i2v_hif4_quant_state.pt}"
OUT="${OUT:-outputs/i2v_hif4_quantized_weights.pt}"
MODE="${MODE:-qdq}"
WEIGHT_DTYPE="${WEIGHT_DTYPE:-bf16}"
DEVICE_ID="${DEVICE_ID:-0}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cuda:${DEVICE_ID}}"
KEEP_FP_REGEX="${KEEP_FP_REGEX:-}"
QUANTIZE_FIRST_LAST_BLOCKS="${QUANTIZE_FIRST_LAST_BLOCKS:-${QUANTIZE_FIRST_LAST_LAYERS:-0}}"
KEEP_FIRST_LAST_LINEAR="${KEEP_FIRST_LAST_LINEAR:-0}"

EXTRA_ARGS=()
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

echo "[export] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[export] Wan device-id=${DEVICE_ID}; QDQ export device=${EXPORT_DEVICE}"
echo "[export] keep first/last Transformer blocks BF16: $([[ "${QUANTIZE_FIRST_LAST_BLOCKS}" == "1" ]] && echo no || echo yes)"
echo "[export] additionally keep first/last Linear modules BF16: $([[ "${KEEP_FIRST_LAST_LINEAR}" == "1" ]] && echo yes || echo no)"

python examples/export_wan22_i2v_quant_weights.py \
  --wan-repo Wan2.2 \
  --ckpt-dir "${CKPT_DIR}" \
  --quant-state "${QUANT_STATE}" \
  --format hif4 \
  --out "${OUT}" \
  --mode "${MODE}" \
  --weight-dtype "${WEIGHT_DTYPE}" \
  --export-device "${EXPORT_DEVICE}" \
  --device-id "${DEVICE_ID}" \
  --t5-cpu \
  "${EXTRA_ARGS[@]}"
