#!/usr/bin/env bash
# implicit 主路径的完整账：逐层 roofline、端到端、逐层计时表、batch 缩放。
# 由外部 flock 包裹（本脚本不加锁）。
set -u
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/libtorch/lib:$HOME/cuda/lib64
DATA=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw
OUT=../results/impl_20261025
mkdir -p "$OUT"
SMID=""

echo "==== nvidia-smi (start) ===="
nvidia-smi --query-gpu=clocks.current.graphics,memory.used,utilization.gpu --format=csv,noheader

echo ""
echo "==== 1) 逐层 roofline：implicit vs (im2col+_int_mm) ===="
./build/bench_layers resnet18_folded.bin calib_minmax.bin 512 15 > "$OUT/bench_layers_b512.txt" 2>&1
grep -E "^(model|合计|占比|总|im2col 加速|GEMM 聚合|理论下限)" "$OUT/bench_layers_b512.txt"
echo "-- implicit 逐层 int32 对照（必须 mismatch 0）--"
grep -E "implicit int32 mismatch" "$OUT/bench_layers_b512.txt" | head -30

echo ""
echo "==== 2) e2e：implicit 默认路径（batch=500，10000 张）===="
USE_IMPLICIT_GEMM=1 IMPLICIT_STRICT=1 ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin \
    "$DATA" --limit 10000 --batch 500 --w-bits 8 --a-bits 8 2>&1 | tee "$OUT/e2e_implicit.txt" | grep -E "implicit|精度|稳态|加速|WARN|FAIL"

echo ""
echo "==== 3) 逐层计时表（PROF_LAYERS=1, batch=500）===="
PROF_LAYERS=1 USE_IMPLICIT_GEMM=1 IMPLICIT_STRICT=1 ./build/gpu_int8 resnet18_folded.bin \
    calib_minmax.bin "$DATA" --limit 500 --batch 500 --w-bits 8 --a-bits 8 \
    > "$OUT/prof_layers_b500.txt" 2>&1
sed -n '/逐层计时/,$p' "$OUT/prof_layers_b500.txt"
grep -E "精度|稳态|加速" "$OUT/prof_layers_b500.txt"

echo ""
echo "==== 4) batch 缩放（implicit，逐层表已按 batch=500 记）===="
for b in 500 1000 1500 2000; do
    printf "batch=%-5s " "$b"
    USE_IMPLICIT_GEMM=1 ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" \
        --limit "$b" --batch "$b" --w-bits 8 --a-bits 8 2>&1 | grep -E "稳态|加速" | tr '\n' ' '
    echo ""
done

echo ""
echo "==== nvidia-smi (end) ===="
nvidia-smi --query-gpu=clocks.current.graphics,memory.used,utilization.gpu --format=csv,noheader
