#!/usr/bin/env bash
# H-048: nsys API counts, normal vs graph.
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h048
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw

nsys profile -t cuda --force-overwrite true -o "$OUT/nsys_graph" \
  env NCNN_CUDA_GRAPH=1 LD_PRELOAD="$HERE/graph_shim.so" \
  "$HERE/ncnn_cuda_graph_eval" "$P" "$M" "$D" --int8 --limit 128 --batch 64 --repeat 1 --warmup 20 \
  > "$OUT/nsys_graph_run.log" 2>&1 || true
nsys stats --report cuda_api_sum "$OUT/nsys_graph.nsys-rep" > "$OUT/nsys_graph_stats.txt" 2>&1 || true

nsys profile -t cuda --force-overwrite true -o "$OUT/nsys_normal" \
  "$HERE/ncnn_cuda_eval_2000c_lazyw8" "$P" "$M" "$D" --int8 --limit 128 --batch 64 --repeat 1 --warmup 20 \
  > "$OUT/nsys_normal_run.log" 2>&1 || true
nsys stats --report cuda_api_sum "$OUT/nsys_normal.nsys-rep" > "$OUT/nsys_normal_stats.txt" 2>&1 || true

echo "=== normal top APIs ==="
grep -E "cudaLaunchKernel|cudaMemcpy|cudaGraphLaunch|Time \(%\)" "$OUT/nsys_normal_stats.txt" | head -8
echo "=== graph top APIs ==="
grep -E "cudaLaunchKernel|cudaMemcpy|cudaGraphLaunch|Time \(%\)" "$OUT/nsys_graph_stats.txt" | head -8
