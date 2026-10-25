#!/usr/bin/env bash
# 一轮跑完：implicit 主路径的端到端 / 逐层计时 / batch 缩放 / 逐层 roofline /
# 多流吞吐 / 旧路径对照。由外部 flock 包裹（本脚本不加锁）。
set -u
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/libtorch/lib:$HOME/cuda/lib64
DATA=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw
OUT=../results/impl_20261025
mkdir -p "$OUT"

echo "==== nvidia-smi (start, idle) ===="
nvidia-smi --query-gpu=clocks.current.graphics,memory.used,utilization.gpu --format=csv,noheader

echo ""
echo "==== A) e2e implicit 默认路径（batch=500，10000 张，W8A8）===="
USE_IMPLICIT_GEMM=1 IMPLICIT_STRICT=1 ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin \
    "$DATA" --limit 10000 --batch 500 --w-bits 8 --a-bits 8 \
    > "$OUT/e2e_implicit.txt" 2>&1
grep -E "\[implicit\]|精度|稳态|加速|WARN|FAIL|CUDA error" "$OUT/e2e_implicit.txt"

echo ""
echo "==== B) 逐层计时表（PROF_LAYERS=1, batch=500）===="
PROF_LAYERS=1 USE_IMPLICIT_GEMM=1 IMPLICIT_STRICT=1 ./build/gpu_int8 resnet18_folded.bin \
    calib_minmax.bin "$DATA" --limit 500 --batch 500 --w-bits 8 --a-bits 8 \
    > "$OUT/prof_layers_b500.txt" 2>&1
sed -n '/逐层计时/,$p' "$OUT/prof_layers_b500.txt"
grep -E "精度|稳态|加速" "$OUT/prof_layers_b500.txt"
nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader > "$OUT/clock_during.txt"

echo ""
echo "==== C) batch 缩放（implicit）===="
for b in 500 1000 1500 2000; do
    printf "batch=%-5s" "$b"
    USE_IMPLICIT_GEMM=1 ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" \
        --limit "$b" --batch "$b" --w-bits 8 --a-bits 8 2>&1 \
        | grep -E "稳态|加速" | tr '\n' ' '
    printf "\n"
done

echo ""
echo "==== D) 逐层 roofline：implicit vs (im2col+_int_mm) ===="
./build/bench_layers resnet18_folded.bin calib_minmax.bin 512 15 \
    > "$OUT/bench_layers_b512.txt" 2>&1
grep -E "^(model|合计|占比|总|im2col 加速|GEMM 聚合|理论下限)" "$OUT/bench_layers_b512.txt"
echo "-- implicit int32 逐层对照（必须 mismatch 0）--"
grep -cE "implicit int32 mismatch: 0 " "$OUT/bench_layers_b512.txt" | xargs echo "mismatch=0 的层数:"
grep -E "implicit int32 mismatch: [^0]" "$OUT/bench_layers_b512.txt" | head -5
echo "-- 融合 epilogue 对照 --"
grep -E "fused epilogue" "$OUT/bench_layers_b512.txt" | head -25

echo ""
echo "==== E) 多流吞吐（跨图并行，按图对齐校验）===="
for s in 1 2 4; do
    echo "--- streams=$s ---"
    ./build/bench_streams resnet18_folded.bin calib_minmax.bin "$DATA" \
        --limit 1000 --streams "$s" --reps 10 --w-bits 8 --a-bits 8 2>&1 \
        | grep -E "顺序|并发|多流加速|正确性|model"
done

echo ""
echo "==== F) 旧路径对照（USE_IMPLICIT_GEMM=0，im2col + _int_mm）===="
USE_IMPLICIT_GEMM=0 ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" \
    --limit 10000 --batch 500 --w-bits 8 --a-bits 8 2>&1 | grep -E "精度|稳态|加速"

echo ""
echo "==== nvidia-smi (end) ===="
nvidia-smi --query-gpu=clocks.current.graphics,memory.used,utilization.gpu --format=csv,noheader
