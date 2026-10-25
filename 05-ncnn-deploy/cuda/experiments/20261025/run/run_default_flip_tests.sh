#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
P=../models/resnet18_cifar10_int8_old.param
M=../models/resnet18_cifar10_int8_old.bin
D=../data_raw/test_batch.raw
OUT=results-h034
mkdir -p "$OUT"

echo "== fusion pairs: default on =="
NCNN_CUDA_FUSE_ACT_DEBUG=1 ./ncnn_cuda_eval_2000c_fusedef "$P" "$M" "$D" --int8 --limit 2 --batch 2 \
  --repeat 1 --warmup 1 --reset-every 1 2>&1 | grep -c "fuse act" || true
echo "== fusion pairs: FUSE_ACT=0 =="
NCNN_CUDA_FUSE_ACT=0 NCNN_CUDA_FUSE_ACT_DEBUG=1 ./ncnn_cuda_eval_2000c_fusedef "$P" "$M" "$D" --int8 --limit 2 --batch 2 \
  --repeat 1 --warmup 1 --reset-every 1 2>&1 | grep -c "fuse act" || true

echo "== 128 logits (default vs a82d8d7 off) =="
./ncnn_cuda_dump_logits_fusedef "$P" "$M" "$D" --int8 --limit 128 --batch 128 --repeat 1 \
  --warmup 20 --reset-every 1 --dump-logits "$OUT/logits_default_128.txt" > "$OUT/run_default_128.log" 2>&1
echo "rc=$?"
md5sum "$OUT/logits_default_128.txt" results-h033/logits_relu_base_128.txt
python3 analyze_logits.py results-h033/logits_relu_base_128.txt "$OUT/logits_default_128.txt"

echo "== 512 logits (default vs a82d8d7) =="
./ncnn_cuda_dump_logits_fusedef "$P" "$M" "$D" --int8 --limit 512 --batch 512 --repeat 1 \
  --warmup 20 --reset-every 1 --dump-logits "$OUT/logits_default_512.txt" > "$OUT/run_default_512.log" 2>&1
echo "rc=$?"
python3 analyze_logits.py results-h031/logits_sync8_512.txt "$OUT/logits_default_512.txt"

echo "== A/B 8 rounds =="
python3 ab_default_flip.py 8 > "$OUT/ab_default_flip.log" 2>&1
tail -6 "$OUT/ab_default_flip.log"
