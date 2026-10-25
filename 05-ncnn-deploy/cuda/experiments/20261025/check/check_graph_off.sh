#!/usr/bin/env bash
# H-048: default-off (env unset) must take the ordinary path unchanged.
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h048
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
"$HERE/ncnn_cuda_graph_eval" "$LAB/models/resnet18_cifar10_int8_old.param" \
  "$LAB/models/resnet18_cifar10_int8_old.bin" "$LAB/data_raw/test_batch.raw" \
  --int8 --limit 128 --batch 64 --repeat 1 --warmup 20 \
  > "$OUT/run_graph_off_128.log" 2>&1
echo "rc=$?"
grep -E "^\[graph\]|^精度|^最优" "$OUT/run_graph_off_128.log"
