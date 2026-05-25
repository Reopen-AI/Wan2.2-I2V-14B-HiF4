#!/usr/bin/env bash
set -euo pipefail

cd /mnt/disk2/home/wujianfeng/com/gcc/nvidia

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HIFLOAT4_GPU_DIR="${HIFLOAT4_GPU_DIR:-/mnt/disk2/home/wujianfeng/com/gcc/HiFloat4/hif4_gpu}"
export PYTHONPATH="${HIFLOAT4_GPU_DIR}:${PYTHONPATH:-}"
export PTQ_WAN22_HIF4_BACKEND="${PTQ_WAN22_HIF4_BACKEND:-official}"
export PTQ_WAN22_HIF4_QDIM="${PTQ_WAN22_HIF4_QDIM:--1}"
export PTQ_WAN22_HIF4_FORCE_FP32="${PTQ_WAN22_HIF4_FORCE_FP32:-0}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
ULYSSES_SIZE="${ULYSSES_SIZE:-${NPROC_PER_NODE}}"
DATASET_ROOT="${DATASET_ROOT:-HiFloat4/datasets}"
OUT_DIR="${OUT_DIR:-outputs/hifloat4_dataset_examples}"
CKPT_DIR="${CKPT_DIR:-/mnt/diskhd/Backup/DownloadModel/Wan2.2-I2V-A14B-BF16/}"
QUANT_STATE="${QUANT_STATE-outputs/i2v_hif4_quant_state.pt}"
LOW_QUANT_WEIGHTS="${LOW_QUANT_WEIGHTS:-}"
HIGH_QUANT_WEIGHTS="${HIGH_QUANT_WEIGHTS:-}"
LIMIT="${LIMIT:-64}"
EVAL_SHARD_INDEX="${EVAL_SHARD_INDEX:-}"
EVAL_NUM_SHARDS="${EVAL_NUM_SHARDS:-}"
FRAME_NUM="${FRAME_NUM:-61}"
SAMPLE_STEPS="${SAMPLE_STEPS:-}"
SIZE="${SIZE:-1280*720}"
CONVERT_MODEL_DTYPE="${CONVERT_MODEL_DTYPE:-1}"
T5_CPU="${T5_CPU:-1}"
OFFLOAD_MODEL="${OFFLOAD_MODEL:-1}"
QUANTIZE_ACT="${QUANTIZE_ACT:-1}"
QUANTIZE_WEIGHT="${QUANTIZE_WEIGHT:-1}"
KEEP_FP_REGEX="${KEEP_FP_REGEX:-}"
QUANTIZE_FIRST_LAST_BLOCKS="${QUANTIZE_FIRST_LAST_BLOCKS:-${QUANTIZE_FIRST_LAST_LAYERS:-0}}"
KEEP_FIRST_LAST_LINEAR="${KEEP_FIRST_LAST_LINEAR:-0}"

EXTRA_ARGS=()
if [[ -n "${LOW_QUANT_WEIGHTS}" || -n "${HIGH_QUANT_WEIGHTS}" ]]; then
  if [[ -z "${LOW_QUANT_WEIGHTS}" || -z "${HIGH_QUANT_WEIGHTS}" ]]; then
    echo "[hifloat4-examples] LOW_QUANT_WEIGHTS and HIGH_QUANT_WEIGHTS must be set together" >&2
    exit 1
  fi
  EXTRA_ARGS+=(--low-quant-weights "${LOW_QUANT_WEIGHTS}")
  EXTRA_ARGS+=(--high-quant-weights "${HIGH_QUANT_WEIGHTS}")
elif [[ -n "${QUANT_STATE}" && "${QUANT_STATE}" != "none" ]]; then
  EXTRA_ARGS+=(--quant-state "${QUANT_STATE}")
fi
if [[ "${CONVERT_MODEL_DTYPE}" != "0" ]]; then
  EXTRA_ARGS+=(--convert-model-dtype)
fi
if [[ "${T5_CPU}" != "0" ]]; then
  EXTRA_ARGS+=(--t5-cpu)
fi
if [[ "${OFFLOAD_MODEL}" != "0" ]]; then
  EXTRA_ARGS+=(--offload-model)
fi
if [[ "${QUANTIZE_ACT}" == "0" ]]; then
  EXTRA_ARGS+=(--no-quant-act)
fi
if [[ "${QUANTIZE_WEIGHT}" == "0" ]]; then
  EXTRA_ARGS+=(--no-quant-weight)
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
if [[ -n "${SAMPLE_STEPS}" ]]; then
  EXTRA_ARGS+=(--sample-steps "${SAMPLE_STEPS}")
fi
if [[ -n "${EVAL_SHARD_INDEX}" ]]; then
  EXTRA_ARGS+=(--eval-shard-index "${EVAL_SHARD_INDEX}")
fi
if [[ -n "${EVAL_NUM_SHARDS}" ]]; then
  EXTRA_ARGS+=(--eval-num-shards "${EVAL_NUM_SHARDS}")
fi

echo "[hifloat4-examples] dataset: ${DATASET_ROOT}"
echo "[hifloat4-examples] output: ${OUT_DIR}"
echo "[hifloat4-examples] limit: ${LIMIT}"
echo "[hifloat4-examples] shard: ${EVAL_SHARD_INDEX:-none}/${EVAL_NUM_SHARDS:-none}"
echo "[hifloat4-examples] frame num: ${FRAME_NUM}; sample steps: ${SAMPLE_STEPS:-default}; size: ${SIZE}"
echo "[hifloat4-examples] keep first/last Transformer blocks BF16: $([[ "${QUANTIZE_FIRST_LAST_BLOCKS}" == "1" ]] && echo no || echo yes)"
echo "[hifloat4-examples] additionally keep first/last Linear modules BF16: $([[ "${KEEP_FIRST_LAST_LINEAR}" == "1" ]] && echo yes || echo no)"
if [[ -n "${LOW_QUANT_WEIGHTS}" || -n "${HIGH_QUANT_WEIGHTS}" ]]; then
  echo "[hifloat4-examples] loading exported low weights: ${LOW_QUANT_WEIGHTS}"
  echo "[hifloat4-examples] loading exported high weights: ${HIGH_QUANT_WEIGHTS}"
elif [[ -n "${QUANT_STATE}" && "${QUANT_STATE}" != "none" ]]; then
  echo "[hifloat4-examples] loading quant state: ${QUANT_STATE}"
else
  echo "[hifloat4-examples] quant state disabled; running unquantized BF16/converted model"
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  examples/generate_hifloat4_dataset_examples.py \
  --wan-repo Wan2.2 \
  --ckpt-dir "${CKPT_DIR}" \
  --dataset-root "${DATASET_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --format hif4 \
  --size "${SIZE}" \
  --limit "${LIMIT}" \
  --frame-num "${FRAME_NUM}" \
  --ulysses-size "${ULYSSES_SIZE}" \
  --skip-existing \
  "${EXTRA_ARGS[@]}"
