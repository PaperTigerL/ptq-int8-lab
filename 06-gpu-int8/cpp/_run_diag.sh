#!/usr/bin/env bash
# 诊断轮：逐层 implicit vs (im2col+gemm) 、融合 epilogue 计时、引擎逐层表。
# 由外部 flock 包裹。
set -u
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/libtorch/lib:$HOME/cuda/lib64
DATA=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw
OUT=../results/impl_20261025
mkdir -p "$OUT"

echo "== nvidia-smi =="
nvidia-smi --query-gpu=clocks.current.graphics,memory.used,utilization.gpu --format=csv,noheader

echo ""
echo "== 逐层 roofline（batch=512, iters=15）=="
./build/bench_layers resnet18_folded.bin calib_minmax.bin 512 15 > "$OUT/bench_layers_b512.txt" 2>&1
grep -E "^(model|#|合计|占比|总|im2col 加速|GEMM 聚合|理论下限)" "$OUT/bench_layers_b512.txt"
echo ""
echo "-- 逐层明细（implicit / implicit+epi / mismatch）--"
grep -E "implicit int32 mismatch|implicit\+epi|fused epilogue|bit-exact|requant bit" "$OUT/bench_layers_b512.txt"

echo ""
echo "== e2e implicit（10000 张, batch=500）=="
USE_IMPLICIT_GEMM=1 IMPLICIT_STRICT=1 ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin \
    "$DATA" --limit 10000 --batch 500 --w-bits 8 --a-bits 8 > "$OUT/e2e_implicit.txt" 2>&1
grep -E "\[implicit\]|精度|稳态|加速|WARN|FAIL|CUDA error" "$OUT/e2e_implicit.txt"

echo ""
echo "== 引擎逐层表（PROF_LAYERS=1，已跳过首个 forward）=="
PROF_LAYERS=1 USE_IMPLICIT_GEMM=1 IMPLICIT_STRICT=1 ./build/gpu_int8 resnet18_folded.bin \
    calib_minmax.bin "$DATA" --limit 500 --batch 500 --w-bits 8 --a-bits 8 \
    > "$OUT/prof_layers_b500.txt" 2>&1
sed -n '/逐层计时/,$p' "$OUT/prof_layers_b500.txt"
grep -E "精度|稳态|加速" "$OUT/prof_layers_b500.txt"
nvidia-smi --query-gpu=clocks.current.graphics,utilization.gpu --format=csv,noheader

echo ""
echo "== 旧路径对照（USE_IMPLICIT_GEMM=0）=="
USE_IMPLICIT_GEMM=0 ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" \
    --limit 10000 --batch 500 --w-bits 8 --a-bits 8 2>&1 | grep -E "精度|稳态|加速"
