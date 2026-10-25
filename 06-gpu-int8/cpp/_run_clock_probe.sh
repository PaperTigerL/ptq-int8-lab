#!/usr/bin/env bash
# 时钟诊断 + 干净的 A/B：跑的同时每 200 ms 采一次频率/温度/功耗。
set -u
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/libtorch/lib:$HOME/cuda/lib64
DATA=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw
OUT=../results/impl_20261025

sample() {
    for i in $(seq 1 200); do
        nvidia-smi --query-gpu=clocks.current.graphics,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader
        sleep 0.2
    done
}

echo "== 时钟采样 + tiled 跑一轮 =="
sample > "$OUT/clocks_tiled.txt" & SP=$!
./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" --limit 4000 --batch 500 \
    --w-bits 8 --a-bits 8 2>&1 | grep -E "稳态|加速|精度" | tee "$OUT/e2e_tiled_4000.txt"
kill $SP 2>/dev/null

echo ""
echo "== 立刻再跑一轮（热）=="
./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" --limit 10000 --batch 500 \
    --w-bits 8 --a-bits 8 2>&1 | grep -E "稳态|加速|精度" | tee "$OUT/e2e_tiled_hot.txt"

echo ""
echo "== A/B 交错：tiled / fast / tiled / fast（同一二进制，只切 IM2COL_TILED）=="
for r in 1 2; do
    printf "round%d tiled  " "$r"
    ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" --limit 2000 --batch 500 2>&1 | grep 稳态
    printf "round%d fast   " "$r"
    IM2COL_TILED=0 ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" --limit 2000 --batch 500 2>&1 | grep 稳态
done

echo ""
echo "== implicit =="
USE_IMPLICIT_GEMM=1 ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" --limit 2000 --batch 500 2>&1 | grep -E "稳态|精度"

echo ""
echo "== 时钟统计（tiled 轮）=="
awk -F, '{gsub(/ /,"",$1); print $1}' "$OUT/clocks_tiled.txt" | sort -n | uniq -c | sort -rn | head -8
echo "--- 温度/功耗范围 ---"
awk -F, '{gsub(/ /,"",$2); gsub(/ /,"",$3); gsub(/ W/,"",$3); if ($2+0>t) t=$2+0; if ($3+0>p) p=$3+0} END {print "max temp=" t " max power=" p}' "$OUT/clocks_tiled.txt"
