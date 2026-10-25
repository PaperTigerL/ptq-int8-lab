#!/usr/bin/env bash
# H-015 acceptance: CudaMat reuse precision + bounded device memory + batch entry.
# Requires ncnn_cuda_eval_h015 (built from ncnn_cuda_eval_h015.cpp against build-h015).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB="/mnt/d/WorkBuddy/ShowBro/lab"
BIN="$HERE/ncnn_cuda_eval_h015"
MODELS="$LAB/05-ncnn-deploy/models"
DATA="$LAB/05-ncnn-deploy/data_raw/test_batch.raw"
OUT="$HERE/results-h015/verify"
mkdir -p "$OUT"

[ -x "$BIN" ] || { echo "FAIL: missing $BIN"; exit 1; }

pass=0
fail=0
note() { printf '%s\n' "$*" | tee -a "$OUT/verify.log"; }
chk() { local ok="$1" msg="$2"; if [ "$ok" -eq 1 ]; then pass=$((pass+1)); note "PASS  $msg"; else fail=$((fail+1)); note "FAIL  $msg"; fi; }

# 1) fp32 + reset-every 1 (reuse), 3 rounds: precision 88.00±0.3 and bounded memory
LOG="$OUT/fp32_reset1_repeat3.log"
"$BIN" "$MODELS/resnet18_cifar10.param" "$MODELS/resnet18_cifar10.bin" "$DATA" \
  --fp32 --limit 200 --batch 64 --repeat 3 --warmup 5 --reset-every 1 > "$LOG" 2>&1
ROUND_P="$(awk '/round [0-9]+:/{for(i=1;i<=NF;i++) if($i=="精度"){gsub(/%/,"",$(i+1)); print $(i+1)}}' "$LOG")"
ROUND_M="$(awk '/显存已用/{for(i=1;i<=NF;i++) if($i=="显存已用"){print $(i+1)}}' "$LOG")"
FINAL_P="$(awk '/^精度 [0-9]+\.[0-9]+%/{v=$2; gsub(/%/,"",v); last=v} END{print last}' "$LOG")"

chk "$(awk -v v="$FINAL_P" 'BEGIN{print (v>=87.7 && v<=88.3)?1:0}')" "fp32 reuse precision ${FINAL_P}% in 87.7..88.3"
ROUNDS_OK=1
for p in $ROUND_P; do awk -v v="$p" 'BEGIN{exit !(v>=87.7 && v<=88.3)}' || ROUNDS_OK=0; done
chk "$ROUNDS_OK" "all rounds precision within 87.7..88.3 (${ROUND_P//$'\n'/,})"

MEMMIN=$(printf '%s\n' "$ROUND_M" | sort -n | head -1)
MEMMAX=$(printf '%s\n' "$ROUND_M" | sort -n | tail -1)
chk "$( [ -n "$MEMMIN" ] && [ -n "$MEMMAX" ] && awk -v a="$MEMMIN" -v b="$MEMMAX" 'BEGIN{print (b-a<=64)?1:0}' )" "device memory bounded: min=${MEMMIN}MB max=${MEMMAX}MB delta<=64MB"

# 2) single-image entry (batch=1) must still work
LOG="$OUT/batch1.log"
"$BIN" "$MODELS/resnet18_cifar10.param" "$MODELS/resnet18_cifar10.bin" "$DATA" \
  --fp32 --limit 16 --batch 1 --repeat 1 --warmup 2 --reset-every 1 > "$LOG" 2>&1
chk 1 "batch=1 single-image entry exit 0"

# 3) batch 64 / 256 entry
for B in 64 256; do
  LOG="$OUT/batch${B}.log"
  "$BIN" "$MODELS/resnet18_cifar10.param" "$MODELS/resnet18_cifar10.bin" "$DATA" \
    --fp32 --limit 256 --batch "$B" --repeat 1 --warmup 5 --reset-every 1 > "$LOG" 2>&1
  P="$(awk '/^精度 [0-9]+\.[0-9]+%/{v=$2; gsub(/%/,"",v); last=v} END{print last}' "$LOG")"
  chk "$(awk -v v="$P" 'BEGIN{print (v>=80)?1:0}')" "batch=$B entry exit 0, precision ${P}%"
done

NOTE="fp32_reuse=${FINAL_P}% mem_min=${MEMMIN}MB mem_max=${MEMMAX}MB"
note "$NOTE"
note "RESULT pass=$pass fail=$fail"
[ "$fail" -eq 0 ]
