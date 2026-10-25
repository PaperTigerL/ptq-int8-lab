#!/usr/bin/env bash
# H-055: nsys API counts for graph default-on (env NCNN_CUDA_GRAPH unset).
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h055
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw

nsys profile -t cuda --force-overwrite true -o "$OUT/nsys_default_on" \
  env LD_PRELOAD="$HERE/graph_shim.so" \
  "$HERE/ncnn_cuda_graph_eval" "$P" "$M" "$D" --int8 --limit 128 --batch 64 --repeat 1 --warmup 20 \
  > "$OUT/nsys_default_on_run.log" 2>&1 || true
nsys stats --report cuda_api_sum "$OUT/nsys_default_on.nsys-rep" > "$OUT/nsys_default_on_stats.txt" 2>&1 || true
grep -E "cudaLaunchKernel|cudaGraphLaunch|cudaMemcpy|Time \(%\)" "$OUT/nsys_default_on_stats.txt" | head -8
grep -E "^\[graph\]|^精度" "$OUT/nsys_default_on_run.log" | head -8
