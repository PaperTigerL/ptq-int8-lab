#!/usr/bin/env bash
# H-071: cumulative-prefix node profile (marginal kernel costs in graph form).
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h071
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw

python3 "$HERE/patch_nodeprof_cum.py"
NCNN_BUILD_DIR=/home/user/ncnn-cuda/build-ptq EVAL_SRC="$HERE/ncnn_cuda_graph_eval.cpp" \
  EVAL_BIN="$HERE/ncnn_cuda_graph_eval" bash "$HERE/build.sh" > "$OUT/build2.log" 2>&1
tail -1 "$OUT/build2.log"

NCNN_CUDA_GRAPH_NODEPROF=1 LD_PRELOAD="$HERE/graph_shim.so" \
  "$HERE/ncnn_cuda_graph_eval" "$P" "$M" "$D" --int8 --limit 8 --batch 8 --repeat 1 --warmup 4 \
  > "$OUT/nodeprof_cum.log" 2>&1 || true
grep -c "^CUM" "$OUT/nodeprof_cum.log" || true
grep -E "^CUM (0|90) " "$OUT/nodeprof_cum.log"
