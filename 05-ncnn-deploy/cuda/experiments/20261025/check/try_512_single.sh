#!/usr/bin/env bash
# H-039: 512 logits in a single formal batch (multi-batch dump copy is known-crashy).
set -u
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h039
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64

for mode in 1 0; do
  env NCNN_CUDA_LAZY_W8=$mode "$HERE/ncnn_cuda_dump_logits_lazyw8" "$P" "$M" "$D" \
    --int8 --limit 512 --batch 512 --repeat 1 --warmup 20 --reset-every 1 \
    --dump-logits "$OUT/logits_512_single_lazy${mode}.txt" > "$OUT/run_512_single_lazy${mode}.log" 2>&1
  echo "lazy=$mode rc=$? lines=$(wc -l < "$OUT/logits_512_single_lazy${mode}.txt" 2>/dev/null || echo 0)"
  tail -1 "$OUT/run_512_single_lazy${mode}.log" | head -c 100; echo
done
echo "== lazy1 vs lazy0 512 =="
python3 "$HERE/analyze_logits.py" "$OUT/logits_512_single_lazy0.txt" "$OUT/logits_512_single_lazy1.txt"
echo "== lazy1 vs cf8fc38 512 =="
python3 "$HERE/analyze_logits.py" "$HERE/results-h035/logits_ipgemv_512.txt" "$OUT/logits_512_single_lazy1.txt"
