#!/usr/bin/env bash
set -euo pipefail

cd /mnt/disk2/home/wujianfeng/com/gcc/nvidia

GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
NUM_SHARDS="${#GPUS[@]}"

echo "[hifloat4-examples-dp] launching ${NUM_SHARDS} standalone workers on GPUs: ${GPU_LIST}"

pids=()
for shard in "${!GPUS[@]}"; do
  gpu="${GPUS[$shard]}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export NPROC_PER_NODE=1
    export ULYSSES_SIZE=1
    export EVAL_SHARD_INDEX="${shard}"
    export EVAL_NUM_SHARDS="${NUM_SHARDS}"
    echo "[hifloat4-examples-dp] worker ${shard}/${NUM_SHARDS} on physical GPU ${gpu}"
    bash hifloat4_generate_examples.sh
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

exit "${failed}"
