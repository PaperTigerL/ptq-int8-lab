#!/usr/bin/env bash
# H-039: narrow the fallback+dump crash at limit 128.
set -u
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h039
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64

echo "=== A: lazy_on fallback, no dump, 128 ==="
env NCNN_CUDA_IMPLICIT=0 NCNN_CUDA_LAZY_W8=1 \
  "$HERE/ncnn_cuda_eval_2000c_lazyw8" "$P" "$M" "$D" --int8 --limit 128 --batch 64 \
  --repeat 1 --warmup 20 > "$OUT/iselA.log" 2>&1
echo "rc=$?"; tail -2 "$OUT/iselA.log"

echo "=== B: lazy_off fallback, dump, 128 ==="
env NCNN_CUDA_IMPLICIT=0 NCNN_CUDA_LAZY_W8=0 \
  "$HERE/ncnn_cuda_dump_logits_lazyw8" "$P" "$M" "$D" --int8 --limit 128 --batch 64 \
  --repeat 1 --warmup 20 --reset-every 1 --dump-logits "$OUT/iselB_dump.txt" > "$OUT/iselB.log" 2>&1
echo "rc=$?"; tail -2 "$OUT/iselB.log"

echo "=== C: lazy_on fallback, dump, 128 ==="
env NCNN_CUDA_IMPLICIT=0 NCNN_CUDA_LAZY_W8=1 \
  "$HERE/ncnn_cuda_dump_logits_lazyw8" "$P" "$M" "$D" --int8 --limit 128 --batch 64 \
  --repeat 1 --warmup 20 --reset-every 1 --dump-logits "$OUT/iselC_dump.txt" > "$OUT/iselC.log" 2>&1
echo "rc=$?"; tail -2 "$OUT/iselC.log"
