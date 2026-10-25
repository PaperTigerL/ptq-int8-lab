#!/usr/bin/env bash
# H-074: graph-form node count + kernel mix, fused vs not.
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h074
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw

NCNN_BUILD_DIR=/home/user/ncnn-cuda/build-ptq EVAL_SRC="$HERE/ncnn_cuda_graph_eval.cpp" \
  EVAL_BIN="$HERE/ncnn_cuda_graph_eval" bash "$HERE/build.sh" > "$OUT/build_graph.log" 2>&1
tail -1 "$OUT/build_graph.log"

for mode in 0 1; do
  NCNN_CUDA_FUSE_QUANT_PACK=$mode NCNN_CUDA_GRAPH_NODEPROF=1 LD_PRELOAD="$HERE/graph_shim.so" \
    "$HERE/ncnn_cuda_graph_eval" "$P" "$M" "$D" --int8 --limit 8 --batch 8 --repeat 1 --warmup 4 \
    > "$OUT/nodeprof_fuse$mode.log" 2>&1 || true
  echo "=== fuse=$mode nodes=$(grep -c '^NODE ' "$OUT/nodeprof_fuse$mode.log") ==="
  grep -E "^CUM 90 |^\[nodeprof\]" "$OUT/nodeprof_fuse$mode.log"
done
