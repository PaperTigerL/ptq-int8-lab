#!/usr/bin/env bash
# im2col A/B：fast vs tiled（同二进制，只切 IM2COL_TILED），外加逐层计时。
set -u
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/libtorch/lib:$HOME/cuda/lib64
DATA=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw
OUT=../results/impl_20261025
mkdir -p "$OUT"

echo "== nvidia-smi =="
nvidia-smi --query-gpu=clocks.current.graphics,utilization.gpu --format=csv,noheader

echo ""
echo "== 预热（把时钟拉起来，再测）=="
for i in 1 2 3; do
  USE_IMPLICIT_GEMM=0 ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" \
      --limit 512 --batch 512 2>&1 | grep -E "稳态" 
done

echo ""
echo "== A) 旧路径 + im2col tiled（默认）=="
USE_IMPLICIT_GEMM=0 ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" \
    --limit 10000 --batch 500 --w-bits 8 --a-bits 8 2>&1 | grep -E "精度|稳态|加速"

echo ""
echo "== B) 旧路径 + im2col fast（IM2COL_TILED=0）=="
IM2COL_TILED=0 USE_IMPLICIT_GEMM=0 ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" \
    --limit 10000 --batch 500 --w-bits 8 --a-bits 8 2>&1 | grep -E "精度|稳态|加速"

echo ""
echo "== C) implicit 路径（对照）=="
USE_IMPLICIT_GEMM=1 ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" \
    --limit 10000 --batch 500 --w-bits 8 --a-bits 8 2>&1 | grep -E "精度|稳态|加速"

echo ""
echo "== D) 逐层：im2col fast / tiled / gemm / implicit =="
./build/bench_layers resnet18_folded.bin calib_minmax.bin 512 20 > "$OUT/bench_layers_tiled.txt" 2>&1
grep -E "^(#|      |[0-9])" "$OUT/bench_layers_tiled.txt" | grep -E "im2col tiled|^[0-9]" | head -60
echo "-- 汇总 --"
grep -E "^合计|^总|^GEMM" "$OUT/bench_layers_tiled.txt"
echo ""
echo "== E) 逐层计时表（im2col tiled 版, batch=500）=="
PROF_LAYERS=1 USE_IMPLICIT_GEMM=0 ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" \
    --limit 500 --batch 500 > "$OUT/prof_legacy_tiled.txt" 2>&1
sed -n '/逐层计时/,$p' "$OUT/prof_legacy_tiled.txt" | head -30
nvidia-smi --query-gpu=clocks.current.graphics,utilization.gpu --format=csv,noheader
