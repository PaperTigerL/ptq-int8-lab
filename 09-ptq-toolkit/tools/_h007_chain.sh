#!/usr/bin/env bash
# H-20261022-007 串行链：R50 Stage0 → R50 I2 → MV2 I2。
# 串行是刻意的：Stage 0 的 median 极差要求 ≤3%，不能与其他重负载并行。
set -u
cd /mnt/d/WorkBuddy/ShowBro/lab/09-ptq-toolkit
OUT=results/h007
mkdir -p "$OUT"
R50=/mnt/d/WorkBuddy/ShowBro/paper-code/ckpts/ckpt_resnet50_r50aug20_ep18.pth
CHAIN="$OUT/chain.log"
: > "$CHAIN"

echo "chain_start=$(date -Is)" >> "$CHAIN"

echo "[1/3] stage0 resnet50" | tee -a "$CHAIN"
TIMEOUT_S=1200 bash tools/_stage0_run.sh resnet50 1,8,32 "$R50" 20 150
rc1=$?
echo "stage0_resnet50_rc=$rc1 end=$(date -Is)" >> "$CHAIN"

echo "[2/3] i2 resnet50" | tee -a "$CHAIN"
TIMEOUT_S=2700 bash tools/_i2_run.sh resnet50 "$R50" 512 256 4
rc2=$?
echo "i2_resnet50_rc=$rc2 end=$(date -Is)" >> "$CHAIN"

echo "[3/3] i2 mobilenet_v2" | tee -a "$CHAIN"
TIMEOUT_S=1800 bash tools/_i2_run.sh mobilenet_v2 "" 512 256 4
rc3=$?
echo "i2_mobilenet_v2_rc=$rc3 end=$(date -Is)" >> "$CHAIN"

echo "chain_end=$(date -Is)" >> "$CHAIN"
echo "rcs=$rc1,$rc2,$rc3" >> "$CHAIN"
echo "rcs=$rc1,$rc2,$rc3"
exit 0
