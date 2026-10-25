#!/usr/bin/env bash
# Per-chunk accuracy cross-check after the pooling optimization.
# Baseline (H-021, 8918acf): correct counts 433/439/432/435 for chunks 0..3.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
BIN="$HERE/ncnn_cuda_eval_2000c"
P="$LAB/models/resnet18_cifar10_int8_old.param"
B="$LAB/models/resnet18_cifar10_int8_old.bin"
OUT="$HERE/results-h024/opt-chunks"
mkdir -p "$OUT"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
for k in 0 1 2 3; do
  log="$OUT/chunk${k}.log"
  "$BIN" "$P" "$B" "$LAB/data_raw/h021/chunk${k}_500.raw" --int8 \
    --limit 500 --batch 64 --repeat 1 --warmup 20 --reset-every 1 > "$log" 2>&1
  rc=$?
  line=$(grep '^精度' "$log" | tail -1)
  echo "chunk${k} rc=$rc ${line}"
done
