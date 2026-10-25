#!/usr/bin/env bash
# CPU baselines for H-20261023-021 (same build-ptq, CPU executor).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
OUT="$HERE/results-h021"
P="$LAB/models/resnet18_cifar10_int8_old.param"
B="$LAB/models/resnet18_cifar10_int8_old.bin"
P32="$LAB/models/resnet18_cifar10.param"
B32="$LAB/models/resnet18_cifar10.bin"
D="$LAB/data_raw/test_batch.raw"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
"$HERE/ncnn_cpu_eval" "$P32" "$B32" "$D" --fp32 --limit 2000 --threads 4 > "$OUT/cpu_fp32_2000.txt" 2>&1
echo "cpu_fp32 rc=$?"
"$HERE/ncnn_cpu_eval" "$P" "$B" "$D" --int8 --limit 2000 --threads 4 > "$OUT/cpu_int8_2000.txt" 2>&1
echo "cpu_int8 rc=$?"
for k in 0 1 2 3; do
  "$HERE/ncnn_cpu_eval" "$P" "$B" "$LAB/data_raw/h021/chunk${k}_500.raw" --int8 --limit 500 --threads 4 \
    > "$OUT/cpu_int8_chunk${k}.txt" 2>&1
  echo "cpu_int8_chunk${k} rc=$?"
done
tail -1 "$OUT/cpu_fp32_2000.txt"
tail -1 "$OUT/cpu_int8_2000.txt"
for k in 0 1 2 3; do tail -1 "$OUT/cpu_int8_chunk${k}.txt"; done
