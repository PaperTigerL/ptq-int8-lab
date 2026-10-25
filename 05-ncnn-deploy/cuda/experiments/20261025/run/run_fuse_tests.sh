#!/usr/bin/env bash
# Minimal activation-fusion tests: baseline, ReLU, Clip, LeakyReLU, no-act.
set -uo pipefail
cd /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
BIN=./ncnn_cuda_dump_logits_fuse
M=../models/resnet18_cifar10_int8_old.bin
D=../data_raw/test_batch.raw
OUT=results-h028
BASE_PARAM=../models/resnet18_cifar10_int8_old.param
CLIP_PARAM=$OUT/resnet18_cifar10_int8_old_clip.param
LEAKY_PARAM=$OUT/resnet18_cifar10_int8_old_leaky.param
mkdir -p "$OUT"

run() {
  local tag="$1"; local param="$2"; local fuse="$3"
  if [ "$fuse" = "1" ]; then
    NCNN_CUDA_FUSE_ACT=1 NCNN_CUDA_FUSE_ACT_DEBUG=1 \
      "$BIN" "$param" "$M" "$D" --int8 --limit 128 --batch 128 \
      --repeat 1 --warmup 20 --reset-every 1 \
      --dump-logits "$OUT/logits_${tag}_128.txt" > "$OUT/run_${tag}_128.log" 2>&1
  else
    "$BIN" "$param" "$M" "$D" --int8 --limit 128 --batch 128 \
      --repeat 1 --warmup 20 --reset-every 1 \
      --dump-logits "$OUT/logits_${tag}_128.txt" > "$OUT/run_${tag}_128.log" 2>&1
  fi
  local rc=$?
  echo "$tag rc=$rc lines=$(wc -l < "$OUT/logits_${tag}_128.txt") fused=$(grep -c 'fuse act' "$OUT/run_${tag}_128.log" || true) $(grep '^精度' "$OUT/run_${tag}_128.log" | tail -1)"
}

run base "$BASE_PARAM" 0
run relu_fused "$BASE_PARAM" 1
run clip_unfused "$CLIP_PARAM" 0
run clip_fused "$CLIP_PARAM" 1
run leaky_unfused "$LEAKY_PARAM" 0
run leaky_fused "$LEAKY_PARAM" 1

echo "== md5 =="
md5sum "$OUT"/logits_base_128.txt "$OUT"/logits_relu_fused_128.txt \
       "$OUT"/logits_clip_unfused_128.txt "$OUT"/logits_clip_fused_128.txt \
       "$OUT"/logits_leaky_unfused_128.txt "$OUT"/logits_leaky_fused_128.txt
