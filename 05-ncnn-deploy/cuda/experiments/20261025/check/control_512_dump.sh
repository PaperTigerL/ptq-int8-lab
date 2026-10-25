#!/usr/bin/env bash
# H-039: is the 512 dump crash specific to lazy, or pre-existing/flaky?
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
  env "$@" "$bin" "$P" "$M" "$D" --int8 --limit 512 --batch 64 --repeat 1 --warmup 20 \
    --reset-every 1 --dump-logits "$OUT/ctl_${tag}.txt" > "$OUT/ctl_${tag}.log" 2>&1
  echo "$tag rc=$? lines=$(wc -l < "$OUT/ctl_${tag}.txt" 2>/dev/null || echo 0)"
  tail -1 "$OUT/ctl_${tag}.log" | head -c 100; echo
}

try old_bin        "$HERE/ncnn_cuda_dump_logits_ipgemv"
try new_bin_off    "$HERE/ncnn_cuda_dump_logits_lazyw8" NCNN_CUDA_LAZY_W8=0
try new_bin_lazy   "$HERE/ncnn_cuda_dump_logits_lazyw8" NCNN_CUDA_LAZY_W8=1
