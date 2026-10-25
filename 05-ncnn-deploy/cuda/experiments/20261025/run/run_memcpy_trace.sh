#!/usr/bin/env bash
# H-039: run 128-image eval under LD_PRELOAD memcpy interposer.
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64

gcc -O2 -shared -fPIC -o "$HERE/memcpy_trace.so" "$HERE/memcpy_trace.c" -ldl
mkdir -p "$HERE/results-h039"

MEMCPY_TRACE_OUT="$HERE/results-h039/memcpy_trace_128.txt" \
LD_PRELOAD="$HERE/memcpy_trace.so" \
"$HERE/ncnn_cuda_eval_2000c_ipgemv" \
  "$LAB/models/resnet18_cifar10_int8_old.param" \
  "$LAB/models/resnet18_cifar10_int8_old.bin" \
  "$LAB/data_raw/test_batch.raw" \
  --int8 --limit 128 --batch 128 --repeat 1 --warmup 20 \
  > "$HERE/results-h039/run_trace_128.log" 2>&1
echo "exit=$?"
tail -20 "$HERE/results-h039/run_trace_128.log"
echo "--- trace totals ---"
grep -E "^TOTAL" "$HERE/results-h039/memcpy_trace_128.txt"
