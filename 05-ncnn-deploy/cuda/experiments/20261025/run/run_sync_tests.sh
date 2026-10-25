#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=../models/resnet18_cifar10_int8_old.param
M=../models/resnet18_cifar10_int8_old.bin
D=../data_raw/test_batch.raw
OUT=results-h031
mkdir -p "$OUT"

echo "== 128 logits: sync-every 8 vs a82d8d7 default =="
./ncnn_cuda_dump_logits_sync "$P" "$M" "$D" --int8 --limit 128 --batch 128 --repeat 1 \
  --warmup 20 --reset-every 1 --sync-every 8 --dump-logits "$OUT/logits_sync8_128.txt" > "$OUT/run_sync8_128.log" 2>&1
echo "rc=$?"
./ncnn_cuda_dump_logits_sync "$P" "$M" "$D" --int8 --limit 128 --batch 128 --repeat 1 \
  --warmup 20 --reset-every 1 --sync-every 1 --dump-logits "$OUT/logits_sync1_128.txt" > "$OUT/run_sync1_128.log" 2>&1
echo "rc=$?"
md5sum "$OUT/logits_sync8_128.txt" "$OUT/logits_sync1_128.txt" results-h030/logits_default_off_128.txt
python3 analyze_logits.py results-h030/logits_default_off_128.txt "$OUT/logits_sync8_128.txt"

echo "== 512 logits: sync-every 8 vs baseline =="
./ncnn_cuda_dump_logits_sync "$P" "$M" "$D" --int8 --limit 512 --batch 512 --repeat 1 \
  --warmup 20 --reset-every 1 --sync-every 8 --dump-logits "$OUT/logits_sync8_512.txt" > "$OUT/run_sync8_512.log" 2>&1
echo "rc=$?"
python3 analyze_logits.py results-h029/logits_pool_512.txt "$OUT/logits_sync8_512.txt"

echo "== A/B 512 (8 rounds) =="
python3 ab_sync_512.py 8 > "$OUT/ab_sync_512.txt" 2>&1
tail -14 "$OUT/ab_sync_512.txt"
