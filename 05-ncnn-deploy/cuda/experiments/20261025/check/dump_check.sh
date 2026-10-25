#!/usr/bin/env bash
# Reproduce the dump-logits crash at different limits.
set -uo pipefail
cd /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
BIN=./ncnn_cuda_eval_2000c_e053c57dump
P=../models/resnet18_cifar10_int8_old.param
M=../models/resnet18_cifar10_int8_old.bin
D=../data_raw/test_batch.raw
mkdir -p results-h024/dumpcheck
for n in 64 100 128; do
  for i in 1 2; do
    rm -f results-h024/dumpcheck/dump_${n}_${i}.txt
    "$BIN" "$P" "$M" "$D" --int8 --limit "$n" --batch 64 --repeat 1 \
      --warmup 20 --reset-every 1 \
      --dump-logits "results-h024/dumpcheck/dump_${n}_${i}.txt" \
      > "results-h024/dumpcheck/log_${n}_${i}.txt" 2>&1
    rc=$?
    lines=$(wc -l < "results-h024/dumpcheck/dump_${n}_${i}.txt")
    acc=$(grep '^精度' "results-h024/dumpcheck/log_${n}_${i}.txt" | tail -1)
    echo "n=$n try=$i rc=$rc lines=$lines $acc"
    if [ "$rc" -ne 0 ]; then tail -3 "results-h024/dumpcheck/log_${n}_${i}.txt"; fi
  done
done
