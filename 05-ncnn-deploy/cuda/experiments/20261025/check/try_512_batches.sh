#!/usr/bin/env bash
# H-039: 512 dump crash appears at ~108 images regardless of binary; try other batch sizes.
set -u
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h039
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64

for b in 16 32 128; do
  env NCNN_CUDA_LAZY_W8=1 "$HERE/ncnn_cuda_dump_logits_lazyw8" "$P" "$M" "$D" \
    --int8 --limit 512 --batch $b --repeat 1 --warmup 20 --reset-every 1 \
    --dump-logits "$OUT/logits_lazy_512_b${b}.txt" > "$OUT/run_lazy_512_b${b}.log" 2>&1
  echo "batch=$b rc=$? lines=$(wc -l < "$OUT/logits_lazy_512_b${b}.txt" 2>/dev/null || echo 0)"
  tail -1 "$OUT/run_lazy_512_b${b}.log" | head -c 100; echo
done
