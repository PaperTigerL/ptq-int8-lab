#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
BIN=./ncnn_cuda_eval_2000c_e053c57dump
P=../models/resnet18_cifar10_int8_old.param
M=../models/resnet18_cifar10_int8_old.bin
D=../data_raw/test_batch.raw
mkdir -p results-h024/dumpcheck

echo "== n=100 no dump, run 1 =="
"$BIN" "$P" "$M" "$D" --int8 --limit 100 --batch 64 --repeat 1 --warmup 20 --reset-every 1 > /tmp/nd1.log 2>&1
echo "rc=$?"; grep '^精度' /tmp/nd1.log; tail -1 /tmp/nd1.log

echo "== n=100 no dump, run 2 =="
"$BIN" "$P" "$M" "$D" --int8 --limit 100 --batch 64 --repeat 1 --warmup 20 --reset-every 1 > /tmp/nd2.log 2>&1
echo "rc=$?"; grep '^精度' /tmp/nd2.log; tail -1 /tmp/nd2.log

echo "== n=100 fallback with dump =="
NCNN_CUDA_IMPLICIT=0 "$BIN" "$P" "$M" "$D" --int8 --limit 100 --batch 64 --repeat 1 --warmup 20 --reset-every 1 --dump-logits results-h024/dumpcheck/fallback_100.txt > /tmp/fd.log 2>&1
echo "rc=$? lines=$(wc -l < results-h024/dumpcheck/fallback_100.txt)"; grep '^精度' /tmp/fd.log; tail -1 /tmp/fd.log

echo "== n=100 dump, batch 128 =="
"$BIN" "$P" "$M" "$D" --int8 --limit 100 --batch 128 --repeat 1 --warmup 20 --reset-every 1 --dump-logits results-h024/dumpcheck/b128_100.txt > /tmp/b128.log 2>&1
echo "rc=$? lines=$(wc -l < results-h024/dumpcheck/b128_100.txt)"; grep '^精度' /tmp/b128.log; tail -1 /tmp/b128.log

echo "== n=200 dump =="
"$BIN" "$P" "$M" "$D" --int8 --limit 200 --batch 64 --repeat 1 --warmup 20 --reset-every 1 --dump-logits results-h024/dumpcheck/dump_200.txt > /tmp/d200.log 2>&1
echo "rc=$? lines=$(wc -l < results-h024/dumpcheck/dump_200.txt)"; grep '^精度' /tmp/d200.log; tail -1 /tmp/d200.log
