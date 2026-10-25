#!/usr/bin/env bash
# 非卷积段微基准 + 交替 A/B（cublas vs 混合），三轮交错，压掉时钟/顺序偏差。
set -u
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/libtorch/lib:$HOME/cuda/lib64
DATA=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw
LOG=../results/ab2_20261025.log
: > "$LOG"

clock() { nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader; }
one() {
  local tag="$1"; shift
  local out
  out=$("$@" 2>&1)
  echo "$out" >> "$LOG"
  local steady=$(echo "$out" | grep -o "稳态 [0-9.]* ms/img" | head -1)
  local acc=$(echo "$out" | grep -o "精度 [0-9.]*%" | head -1)
  local impl=$(echo "$out" | grep -o "按层混合：.*层）" | head -1)
  echo "$tag | $steady | $acc | clock $(clock) | $impl"
}

echo "==== probe_overhead ===="
./build/probe_overhead 200 | tee -a "$LOG"
echo ""
echo "==== alternating A/B (batch=512, limit=512, 100 预热 + 60 取最快) ===="
for r in 1 2 3; do
  one "R$r cublas " env -u LAYER_IMPL -u LAYER_IMPL_JSON USE_IMPLICIT_GEMM=0 \
      ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" --limit 512 --batch 512
  one "R$r mixed  " env -u USE_IMPLICIT_GEMM -u LAYER_IMPL -u LAYER_IMPL_JSON \
      ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" --limit 512 --batch 512
done

echo ""
echo "==== clock final: $(clock) ===="
