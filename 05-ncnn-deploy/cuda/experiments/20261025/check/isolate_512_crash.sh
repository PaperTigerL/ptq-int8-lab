#!/usr/bin/env bash
# H-055: isolate the 512/warmup30 crash: old binary vs graph binary off, warmup 20/30.
set -u
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h055
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64

try() {
  local tag=$1 bin=$2 warm=$3; shift 3
  env "$@" "$bin" "$P" "$M" "$D" --int8 --limit 512 --batch 64 --repeat 1 --warmup "$warm" \
    > "$OUT/iso_${tag}.log" 2>&1
  echo "$tag rc=$?"; tail -1 "$OUT/iso_${tag}.log" | head -c 100; echo
}

try old_w30 "$HERE/ncnn_cuda_eval_2000c_lazyw8" 30
try old_w20 "$HERE/ncnn_cuda_eval_2000c_lazyw8" 20
try off_w30 "$HERE/ncnn_cuda_graph_eval" 30 NCNN_CUDA_GRAPH=0
try off_w20 "$HERE/ncnn_cuda_graph_eval" 20 NCNN_CUDA_GRAPH=0
