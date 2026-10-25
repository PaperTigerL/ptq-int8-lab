#!/usr/bin/env bash
# H-039: fallback dump at smaller limits (128 crashes even with lazy off).
set -u
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h039
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64

for n in 32 64; do
  for mode in 0 1; do
    tag="fb${n}_lazy${mode}"
    env NCNN_CUDA_IMPLICIT=0 NCNN_CUDA_LAZY_W8=$mode \
      "$HERE/ncnn_cuda_dump_logits_lazyw8" "$P" "$M" "$D" --int8 --limit $n --batch 32 \
      --repeat 1 --warmup 10 --reset-every 1 --dump-logits "$OUT/logits_${tag}.txt" \
      > "$OUT/run_logits_${tag}.log" 2>&1
    echo "$tag rc=$? lines=$(wc -l < "$OUT/logits_${tag}.txt" 2>/dev/null || echo 0)"
    tail -1 "$OUT/run_logits_${tag}.log" | head -c 120; echo
  done
  python3 "$HERE/analyze_logits.py" "$OUT/logits_fb${n}_lazy0.txt" "$OUT/logits_fb${n}_lazy1.txt" 2>&1 | tail -4
done
