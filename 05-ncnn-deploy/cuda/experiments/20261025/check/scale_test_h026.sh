#!/usr/bin/env bash
# Scale-cache invalidation test: same process, mutate layer-1 bottom scale.
set -uo pipefail
cd /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64
BIN=./ncnn_cuda_scale_test
P=../models/resnet18_cifar10_int8_old.param
M=../models/resnet18_cifar10_int8_old.bin
D=../data_raw/test_batch.raw
OUT=results-h026
mkdir -p "$OUT"

run() {
  local tag="$1"; local when="$2"; local factor="$3"
  local args=()
  if [ -n "$when" ]; then
    args+=(--set-scale-layer 1 --set-scale-factor "$factor" --set-scale-when "$when")
  fi
  "$BIN" "$P" "$M" "$D" --int8 --limit 64 --batch 64 --repeat 1 \
    --warmup 20 --reset-every 1 "${args[@]}" \
    --dump-logits "$OUT/scale_${tag}_64.txt" > "$OUT/scale_${tag}.log" 2>&1
  echo "$tag rc=$? lines=$(wc -l < "$OUT/scale_${tag}_64.txt") $(grep -h 'scale mutate' "$OUT/scale_${tag}.log")"
}

run plain "" 1.0
run ref before 1.05
run test after 1.05
run test_identity after 1.0

echo "== md5 =="
md5sum "$OUT"/scale_plain_64.txt "$OUT"/scale_ref_64.txt "$OUT"/scale_test_64.txt "$OUT"/scale_test_identity_64.txt
