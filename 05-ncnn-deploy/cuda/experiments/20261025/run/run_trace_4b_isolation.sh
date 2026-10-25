#!/usr/bin/env bash
# H-039: isolate when 4-byte scalar H2D happens (warmup vs formal batch).
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64

for pair in "4 4" "4 0"; do
  set -- $pair
  LIM=$1; WARM=$2
  MEMCPY_TRACE_OUT="$HERE/results-h039/memcpy_trace_l${LIM}_w${WARM}.txt" \
  LD_PRELOAD="$HERE/memcpy_trace.so" \
  "$HERE/ncnn_cuda_eval_2000c_lazyw8" \
    "$LAB/models/resnet18_cifar10_int8_old.param" \
    "$LAB/models/resnet18_cifar10_int8_old.bin" \
    "$LAB/data_raw/test_batch.raw" \
    --int8 --limit "$LIM" --batch 128 --repeat 1 --warmup "$WARM" \
    > "$HERE/results-h039/run_l${LIM}_w${WARM}.log" 2>&1
  echo "--- limit=$LIM warmup=$WARM ---"
  grep -E "^TOTAL kind=1" "$HERE/results-h039/memcpy_trace_l${LIM}_w${WARM}.txt"
  grep -E "^1 4 " "$HERE/results-h039/memcpy_trace_l${LIM}_w${WARM}.txt" | awk '{s+=$3; print "  4B site:", $3, "calls"} END{print "  4B total:", s}'
done
