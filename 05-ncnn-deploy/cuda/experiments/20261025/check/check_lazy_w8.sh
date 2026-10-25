#!/usr/bin/env bash
# H-039: logits / top1 / top5 checks for lazy weight upload (opt-in).
set -eu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
HERE=$LAB/cuda
OUT=$HERE/results-h039
P=$LAB/models/resnet18_cifar10_int8_old.param
M=$LAB/models/resnet18_cifar10_int8_old.bin
D=$LAB/data_raw/test_batch.raw
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
mkdir -p "$OUT"

run() {
  local tag=$1 limit=$2; shift 2
  env "$@" "$HERE/ncnn_cuda_dump_logits_lazyw8" "$P" "$M" "$D" \
    --int8 --limit "$limit" --batch 64 --repeat 1 --warmup 20 --reset-every 1 \
    --dump-logits "$OUT/logits_${tag}.txt" > "$OUT/run_logits_${tag}.log" 2>&1
  grep -E "^精度" "$OUT/run_logits_${tag}.log" | tail -1
}

run lazy_128 128 NCNN_CUDA_LAZY_W8=1
run lazy_fb_128 128 NCNN_CUDA_LAZY_W8=1 NCNN_CUDA_IMPLICIT=0
run lazy_512 512 NCNN_CUDA_LAZY_W8=1

echo "== lazy_128 vs cf8fc38 128 =="
python3 "$HERE/analyze_logits.py" "$HERE/results-h035/logits_ipgemv_128.txt" "$OUT/logits_lazy_128.txt"
echo "== lazy_fallback_128 vs cf8fc38 128 =="
python3 "$HERE/analyze_logits.py" "$HERE/results-h035/logits_ipgemv_128.txt" "$OUT/logits_lazy_fb_128.txt"
echo "== lazy_512 vs cf8fc38 512 =="
python3 "$HERE/analyze_logits.py" "$HERE/results-h035/logits_ipgemv_512.txt" "$OUT/logits_lazy_512.txt"
