#!/usr/bin/env bash
# H-039: trace memcpy with lazy weight upload enabled.
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64

MEMCPY_TRACE_OUT="$HERE/results-h039/memcpy_trace_lazy_128.txt" \
NCNN_CUDA_LAZY_W8=1 \
LD_PRELOAD="$HERE/memcpy_trace.so" \
"$HERE/ncnn_cuda_eval_2000c_lazyw8" \
  "$LAB/models/resnet18_cifar10_int8_old.param" \
  "$LAB/models/resnet18_cifar10_int8_old.bin" \
  "$LAB/data_raw/test_batch.raw" \
  --int8 --limit 128 --batch 128 --repeat 1 --warmup 20 \
  > "$HERE/results-h039/run_trace_lazy_128.log" 2>&1
grep -E "^精度|^最优" "$HERE/results-h039/run_trace_lazy_128.log"
echo "--- totals lazy ---"
grep -E "^TOTAL" "$HERE/results-h039/memcpy_trace_lazy_128.txt"
