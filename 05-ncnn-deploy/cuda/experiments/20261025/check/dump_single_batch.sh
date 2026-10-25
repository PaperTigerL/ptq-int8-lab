#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
BIN=./ncnn_cuda_dump_logits_e053c57
P=../models/resnet18_cifar10_int8_old.param
M=../models/resnet18_cifar10_int8_old.bin
D=../data_raw/test_batch.raw
mkdir -p results-h024/logits
for cfg in "256 256" "512 512" "256 256" "512 512"; do
  set -- $cfg
  n=$1; b=$2
  out="results-h024/logits/old_${n}_b${b}_$(date +%H%M%S).txt"
  "$BIN" "$P" "$M" "$D" --int8 --limit "$n" --batch "$b" --repeat 1 --warmup 20 --reset-every 1 --dump-logits "$out" > /tmp/ld.log 2>&1
  rc=$?
  echo "n=$n b=$b rc=$rc lines=$(wc -l < "$out") md5=$(md5sum "$out" | cut -c1-12)"
  if [ "$rc" -ne 0 ]; then tail -2 /tmp/ld.log; fi
done
