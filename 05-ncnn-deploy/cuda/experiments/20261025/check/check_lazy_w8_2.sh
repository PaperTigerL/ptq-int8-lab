#!/usr/bin/env bash
# H-039: finish 512 with lazy on; also prove default path of the new binary is
# bit-identical to the cf8fc38 dump binary (env off).
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h039
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64

env "$HERE/ncnn_cuda_dump_logits_lazyw8" "$P" "$M" "$D" --int8 --limit 128 --batch 64 \
  --repeat 1 --warmup 20 --reset-every 1 --dump-logits "$OUT/logits_default_128.txt" \
  > "$OUT/run_logits_default_128.log" 2>&1
echo "== new binary default vs cf8fc38 128 =="
python3 "$HERE/analyze_logits.py" "$HERE/results-h035/logits_ipgemv_128.txt" "$OUT/logits_default_128.txt"

env NCNN_CUDA_LAZY_W8=1 "$HERE/ncnn_cuda_dump_logits_lazyw8" "$P" "$M" "$D" --int8 --limit 512 --batch 64 \
  --repeat 1 --warmup 20 --reset-every 1 --dump-logits "$OUT/logits_lazy_512.txt" \
  > "$OUT/run_logits_lazy_512.log" 2>&1
echo "== lazy_512 vs cf8fc38 512 =="
python3 "$HERE/analyze_logits.py" "$HERE/results-h035/logits_ipgemv_512.txt" "$OUT/logits_lazy_512.txt"
echo "== lazy_512 self top1/top5 =="
python3 "$HERE/analyze_logits.py" "$OUT/logits_lazy_512.txt"
