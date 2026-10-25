#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=../models/resnet18_cifar10_int8_old.param
M=../models/resnet18_cifar10_int8_old.bin
D=../data_raw/test_batch.raw
OUT=results-h030
for r in 1 2 3; do
  ./ncnn_cuda_profile_layers_pool "$P" "$M" "$D" --int8 --limit 512 --batch 64 \
    --repeat 1 --warmup 20 --reset-every 1 > "$OUT/prof_base_r${r}.log" 2>&1
  ./ncnn_cuda_profile_layers_binop "$P" "$M" "$D" --int8 --limit 512 --batch 64 \
    --repeat 1 --warmup 20 --reset-every 1 > "$OUT/prof_vec_r${r}.log" 2>&1
done
echo "BASE"
grep "TYPE BinaryOp" "$OUT"/prof_base_r*.log
echo "VEC"
grep "TYPE BinaryOp" "$OUT"/prof_vec_r*.log
