#!/usr/bin/env bash
# 基线账：bench_layers 逐层 + 端到端（默认路 / implicit 路）。
# 由外部 flock 包裹，本脚本不自己加锁。
set -u
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/libtorch/lib:$HOME/cuda/lib64
DATA=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw
OUT=../results/impl_20261025
mkdir -p "$OUT"

echo "==== nvidia-smi (start) ===="
nvidia-smi --query-gpu=clocks.current.graphics,memory.used,utilization.gpu --format=csv,noheader

echo "==== build bench_layers ===="
bash build_bench_layers.sh 2>&1 | tail -2

echo "==== bench_layers batch=512 iters=15 ===="
./build/bench_layers resnet18_folded.bin calib_minmax.bin 512 15 2>&1 | tee "$OUT/bench_layers_b512.txt" | grep -E "^(合计|占比|总|im2col 加速|GEMM|理论|model)|implicit int32|fused epilogue|im2col vec|requant bit" | head -80

echo "==== e2e default (im2col + _int_mm) ===="
./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" --limit 10000 --batch 500 --w-bits 8 --a-bits 8 2>&1 | tail -5 | tee "$OUT/e2e_default.txt"

echo "==== e2e USE_IMPLICIT_GEMM=1 ===="
USE_IMPLICIT_GEMM=1 ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" --limit 10000 --batch 500 --w-bits 8 --a-bits 8 2>&1 | tee "$OUT/e2e_implicit.txt" | tail -25

echo "==== nvidia-smi (end) ===="
nvidia-smi --query-gpu=clocks.current.graphics,memory.used,utilization.gpu --format=csv,noheader
