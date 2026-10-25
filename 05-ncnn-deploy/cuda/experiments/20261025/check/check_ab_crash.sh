#!/usr/bin/env bash
# H-055: reproduce A/B crash: run off and on once each at 512.
set -u
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h055
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64

NCNN_CUDA_GRAPH=0 "$HERE/ncnn_cuda_graph_eval" "$P" "$M" "$D" --int8 --limit 512 \
  --batch 64 --repeat 1 --warmup 30 > "$OUT/abcheck_off.log" 2>&1
echo "off rc=$?"; tail -2 "$OUT/abcheck_off.log"

LD_PRELOAD="$HERE/graph_shim.so" "$HERE/ncnn_cuda_graph_eval" "$P" "$M" "$D" --int8 --limit 512 \
  --batch 64 --repeat 1 --warmup 30 > "$OUT/abcheck_on.log" 2>&1
echo "on rc=$?"; tail -2 "$OUT/abcheck_on.log"
