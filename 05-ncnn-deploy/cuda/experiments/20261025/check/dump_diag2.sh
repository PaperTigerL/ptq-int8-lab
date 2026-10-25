#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=../models/resnet18_cifar10_int8_old.param
M=../models/resnet18_cifar10_int8_old.bin
D=../data_raw/test_batch.raw

run() {
  local label="$1"; shift
  "$@" > /tmp/run_$label.log 2>&1
  local rc=$?
  echo "$label rc=$rc $(grep '^精度' /tmp/run_$label.log | tail -1)"
  if [ "$rc" -ne 0 ]; then tail -2 /tmp/run_$label.log; fi
}

echo "== original binary, 100 b64 =="
run orig100 ./ncnn_cuda_eval_2000c "$P" "$M" "$D" --int8 --limit 100 --batch 64 --repeat 1 --warmup 20 --reset-every 1
echo "== original binary, 2000 b64 =="
run orig2000 ./ncnn_cuda_eval_2000c "$P" "$M" "$D" --int8 --limit 2000 --batch 64 --repeat 1 --warmup 20 --reset-every 1
echo "== new binary no dump, 100 b32 =="
run new100b32 ./ncnn_cuda_eval_2000c_e053c57dump "$P" "$M" "$D" --int8 --limit 100 --batch 32 --repeat 1 --warmup 20 --reset-every 1
echo "== new binary no dump, 100 b128 =="
run new100b128 ./ncnn_cuda_eval_2000c_e053c57dump "$P" "$M" "$D" --int8 --limit 100 --batch 128 --repeat 1 --warmup 20 --reset-every 1
echo "== new binary dump, 128 b64 =="
run new128dump ./ncnn_cuda_eval_2000c_e053c57dump "$P" "$M" "$D" --int8 --limit 128 --batch 64 --repeat 1 --warmup 20 --reset-every 1 --dump-logits results-h024/dumpcheck/new_128.txt
echo "== new binary dump, 100 b128 =="
run new100b128dump ./ncnn_cuda_eval_2000c_e053c57dump "$P" "$M" "$D" --int8 --limit 100 --batch 128 --repeat 1 --warmup 20 --reset-every 1 --dump-logits results-h024/dumpcheck/new_100_b128.txt
