#!/usr/bin/env bash
# H-039: isolate fallback crash: baseline vs lazy binary, lazy env on/off.
set -u
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h039
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64

try() {
  local tag=$1 bin=$2; shift 2
  env "$@" "$bin" "$P" "$M" "$D" --int8 --limit 8 --batch 8 --repeat 1 --warmup 2 \
    > "$OUT/isel_${tag}.log" 2>&1
  local rc=$?
  echo "=== $tag rc=$rc ==="
  tail -3 "$OUT/isel_${tag}.log"
}

try base_fb        "$HERE/ncnn_cuda_eval_2000c_ipgemv" NCNN_CUDA_IMPLICIT=0
try lazy_off_fb    "$HERE/ncnn_cuda_eval_2000c_lazyw8" NCNN_CUDA_IMPLICIT=0 NCNN_CUDA_LAZY_W8=0
try lazy_on_fb     "$HERE/ncnn_cuda_eval_2000c_lazyw8" NCNN_CUDA_IMPLICIT=0 NCNN_CUDA_LAZY_W8=1
try lazy_on_impl   "$HERE/ncnn_cuda_eval_2000c_lazyw8" NCNN_CUDA_LAZY_W8=1
