#!/usr/bin/env bash
# H-071: graph-form per-node kernel profile.
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h071
mkdir -p "$OUT"
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw

python3 "$HERE/add_nodeprof.py"
gcc -O2 -shared -fPIC -o "$HERE/graph_shim.so" "$HERE/graph_shim.c" -ldl
NCNN_BUILD_DIR=/home/user/ncnn-cuda/build-ptq EVAL_SRC="$HERE/ncnn_cuda_graph_eval.cpp" \
  EVAL_BIN="$HERE/ncnn_cuda_graph_eval" bash "$HERE/build.sh" > "$OUT/build.log" 2>&1
tail -1 "$OUT/build.log"

NCNN_CUDA_GRAPH_NODEPROF=1 LD_PRELOAD="$HERE/graph_shim.so" \
  "$HERE/ncnn_cuda_graph_eval" "$P" "$M" "$D" --int8 --limit 8 --batch 8 --repeat 1 --warmup 4 \
  > "$OUT/nodeprof.log" 2>&1 || true
grep -c "^NODEPROF" "$OUT/nodeprof.log" || true
grep -E "^\[graph\]|^\[nodeprof\]" "$OUT/nodeprof.log" | head -6
head -8 "$OUT/nodeprof.log"
