#!/usr/bin/env bash
# H-055: gdb backtrace for graph-off 512/batch64 heap corruption.
set -u
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h055
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64

gdb --batch -ex "set env NCNN_CUDA_GRAPH 0" -ex run -ex "bt 15" --args \
  "$HERE/ncnn_cuda_graph_eval" "$P" "$M" "$D" --int8 --limit 512 --batch 64 --repeat 1 --warmup 30 \
  > "$OUT/gdb_off512.log" 2>&1 || true
grep -A20 "Program received\|malloc" "$OUT/gdb_off512.log" | head -30
