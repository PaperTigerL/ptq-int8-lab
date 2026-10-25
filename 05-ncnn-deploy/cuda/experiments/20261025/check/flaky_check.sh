#!/usr/bin/env bash
# H-055: estimate flakiness of the 512 crash for old vs graph(with off) binaries.
set -u
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h055
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64

for i in 1 2 3; do
  "$HERE/ncnn_cuda_eval_2000c_lazyw8" "$P" "$M" "$D" --int8 --limit 512 --batch 64 \
    --repeat 1 --warmup 30 > "$OUT/flaky_old_$i.log" 2>&1
  echo "old_$i rc=$?"
  NCNN_CUDA_GRAPH=0 "$HERE/ncnn_cuda_graph_eval" "$P" "$M" "$D" --int8 --limit 512 --batch 64 \
    --repeat 1 --warmup 30 > "$OUT/flaky_goff_$i.log" 2>&1
  echo "graph_off_$i rc=$?"
done
