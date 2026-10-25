#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=../models/resnet18_cifar10_int8_old.param
M=../models/resnet18_cifar10_int8_old.bin
D=../data_raw/test_batch.raw
OUT=results-h032
mkdir -p "$OUT"
BIN=./ncnn_cuda_ab_harness

"$BIN" "$P" "$M" "$D" --int8 --ab-switch pool --ab-images 256 --ab-batch 32 --ab-rounds 8 --ab-warmup 30 > "$OUT/ab_pool.log" 2>&1
echo "pool_rc=$?"
"$BIN" "$P" "$M" "$D" --int8 --ab-switch binop --ab-images 256 --ab-batch 32 --ab-rounds 8 --ab-warmup 30 > "$OUT/ab_binop.log" 2>&1
echo "binop_rc=$?"
"$BIN" "$P" "$M" "$D" --int8 --ab-switch relu --ab-images 256 --ab-batch 32 --ab-rounds 8 --ab-warmup 30 > "$OUT/ab_relu.log" 2>&1
echo "relu_rc=$?"

for f in "$OUT/ab_pool.log" "$OUT/ab_binop.log" "$OUT/ab_relu.log"; do
  echo "===== $f ====="
  grep -E "A/B switch|relu fuse pairs|round |SUMMARY|DECISION|致命|arena 溢出" "$f" || tail -3 "$f"
done
