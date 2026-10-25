#!/usr/bin/env bash
# H-079: graph node profile with border-aware conv (env on).
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h079
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw

NCNN_BUILD_DIR=/home/user/ncnn-cuda/build-ptq EVAL_SRC="$HERE/ncnn_cuda_graph_eval.cpp" \
  EVAL_BIN="$HERE/ncnn_cuda_graph_eval" bash "$HERE/build.sh" > "$OUT/build_graph.log" 2>&1
tail -1 "$OUT/build_graph.log"

NCNN_CUDA_CONV_BORDER=1 NCNN_CUDA_GRAPH_NODEPROF=1 LD_PRELOAD="$HERE/graph_shim.so" \
  "$HERE/ncnn_cuda_graph_eval" "$P" "$M" "$D" --int8 --limit 8 --batch 8 --repeat 1 --warmup 4 \
  > "$OUT/nodeprof_border.log" 2>&1 || true
echo "nodes=$(grep -c '^NODE ' "$OUT/nodeprof_border.log")"
grep -E "^CUM (0|70) " "$OUT/nodeprof_border.log"
cp "$HERE/map_nodeprof_cum.py" /tmp/map_bd.py
sed -i 's#results-h071/nodeprof_cum.log#results-h079/nodeprof_border.log#' /tmp/map_bd.py
python3 /tmp/map_bd.py | head -12
