#!/usr/bin/env bash
# H-048: build the CUDA Graph probe and run it once on 8 images.
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h048
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64

gcc -O2 -shared -fPIC -o "$HERE/graph_shim.so" "$HERE/graph_shim.c" -ldl
NCNN_BUILD_DIR=/home/user/ncnn-cuda/build-ptq \
  EVAL_SRC="$HERE/ncnn_cuda_graph_eval.cpp" EVAL_BIN="$HERE/ncnn_cuda_graph_eval" \
  bash "$HERE/build.sh" > "$OUT/build_graph_eval.log" 2>&1
tail -2 "$OUT/build_graph_eval.log"

NCNN_CUDA_GRAPH=1 LD_PRELOAD="$HERE/graph_shim.so" \
  "$HERE/ncnn_cuda_graph_eval" \
  "$LAB/models/resnet18_cifar10_int8_old.param" \
  "$LAB/models/resnet18_cifar10_int8_old.bin" \
  "$LAB/data_raw/test_batch.raw" \
  --int8 --limit 8 --batch 8 --repeat 1 --warmup 2 \
  > "$OUT/graph_probe_8.log" 2>&1 || true
cat "$OUT/graph_probe_8.log"
