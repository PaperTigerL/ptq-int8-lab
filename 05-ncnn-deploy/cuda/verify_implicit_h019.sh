#!/usr/bin/env bash
# H-019: default implicit int8 path + NCNN_CUDA_IMPLICIT=0 fallback regression.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB="/mnt/d/WorkBuddy/ShowBro/lab"
BIN="$HERE/ncnn_cuda_eval_h019"
MODELS="$LAB/05-ncnn-deploy/models"
DATA="$LAB/05-ncnn-deploy/data_raw/test_batch.raw"
OUT="$HERE/results-h019/verify"
mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "FAIL missing $BIN"; exit 1; }

P=0; F=0
note() { printf '%s\n' "$*" | tee -a "$OUT/verify.log"; }
chk() { local ok="$1" msg="$2"; if [ "$ok" -eq 1 ]; then P=$((P+1)); note "PASS  $msg"; else F=$((F+1)); note "FAIL  $msg"; fi; }
precision() { awk '/^精度 [0-9]+\.[0-9]+%/{v=$2; gsub(/%/,"",v); last=v} END{print last}' "$1"; }

unset NCNN_CUDA_IMPLICIT
DEF="$OUT/default.log"
"$BIN" "$MODELS/resnet18_cifar10_int8_old.param" "$MODELS/resnet18_cifar10_int8_old.bin" "$DATA" \
  --int8 --limit 100 --batch 64 --repeat 1 --warmup 5 --no-reset > "$DEF" 2>&1
DP="$(precision "$DEF")"
chk "$(awk -v v="$DP" 'BEGIN{print (v>=88.7 && v<=89.3)?1:0}')" "default implicit precision ${DP}% in 88.7..89.3"

OLD="$OUT/old_path.log"
NCNN_CUDA_IMPLICIT=0 "$BIN" "$MODELS/resnet18_cifar10_int8_old.param" "$MODELS/resnet18_cifar10_int8_old.bin" "$DATA" \
  --int8 --limit 100 --batch 64 --repeat 1 --warmup 5 --no-reset > "$OLD" 2>&1
OP="$(precision "$OLD")"
chk "$(awk -v v="$OP" 'BEGIN{print (v>=88.7 && v<=89.3)?1:0}')" "env=0 old path precision ${OP}% in 88.7..89.3"

for B in 64 256; do
  L="$OUT/batch${B}.log"
  "$BIN" "$MODELS/resnet18_cifar10_int8_old.param" "$MODELS/resnet18_cifar10_int8_old.bin" "$DATA" \
    --int8 --limit 256 --batch "$B" --repeat 1 --warmup 5 --no-reset > "$L" 2>&1
  BP="$(precision "$L")"
  chk "$(awk -v v="$BP" 'BEGIN{print (v>=80)?1:0}')" "default implicit batch=$B exit0, precision ${BP}%"
done

note "default=${DP}% old=${OP}%"
note "RESULT pass=$P fail=$F"
[ "$F" -eq 0 ]
