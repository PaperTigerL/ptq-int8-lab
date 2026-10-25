#!/usr/bin/env bash
# I2 运行器：CPU-only，记录 nvidia-smi/free/dmesg 前后对照、退出码与超时。
# 用法：_i2_run.sh <resnet50|mobilenet_v2> [weights] [calib-n] [batch-size] [threads]
# 环境：TIMEOUT_S（默认 2700 秒）
set -u
cd /mnt/d/WorkBuddy/ShowBro/lab/09-ptq-toolkit
mkdir -p results/i2
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=""

SMI=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)
MODEL="${1:-resnet50}"
WEIGHTS="${2:-}"
CALIB="${3:-512}"
BATCH="${4:-256}"
THREADS="${5:-4}"
TIMEOUT_S="${TIMEOUT_S:-2700}"
TAG="${TAG:-}"
PIN_CPUS="${PIN_CPUS:-}"
SETTLE_S="${SETTLE_S:-0}"

LOG="results/i2/layer_sensitivity_${MODEL}${TAG}.log"
STATUS="results/i2/layer_sensitivity_${MODEL}${TAG}.status"

snapshot() {
  local tag="$1"
  {
    echo "== $tag @ $(date -Is) =="
    "$SMI" || true
    echo "== $tag compute apps =="
    "$SMI" --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader || true
    echo "== $tag free -m =="
    free -m || true
    echo "== $tag dmesg tail =="
    dmesg 2>/dev/null | tail -30 || echo "(dmesg unavailable)"
  } > "results/i2/nvidia_smi_${tag}_${MODEL}${TAG}.txt" 2>&1
}

: > "$LOG"
echo "start=$(date -Is) pid=$$ timeout_s=$TIMEOUT_S" >> "$LOG"
echo "model=$MODEL calib=$CALIB batch=$BATCH threads=$THREADS \
weights=${WEIGHTS:-(default)} pin=${PIN_CPUS:-(none)} settle=${SETTLE_S}s" >> "$LOG"
snapshot before

if [ "$SETTLE_S" -gt 0 ]; then
  sleep "$SETTLE_S"
fi

EXTRA=()
if [ -n "$WEIGHTS" ]; then EXTRA=(--weights "$WEIGHTS"); fi
PREFIX=()
if [ -n "$PIN_CPUS" ]; then PREFIX=(taskset -c "$PIN_CPUS"); fi

START_EPOCH=$(date +%s)
stdbuf -oL -eL timeout --signal=TERM "$TIMEOUT_S" \
  "${PREFIX[@]}" /home/user/pyenv-torch/bin/python -u tools/layer_sensitivity.py \
  --model "$MODEL" --calib-n "$CALIB" --batch-size "$BATCH" \
  --threads "$THREADS" --rank-n 64 --recheck-n 512 \
  --recheck-top-w4 10 --recheck-top-w8 6 "${EXTRA[@]}" \
  >> "$LOG" 2>&1
rc=$?
END_EPOCH=$(date +%s)
DURATION=$((END_EPOCH - START_EPOCH))

echo "exit_code=$rc" >> "$LOG"
echo "duration_s=$DURATION" >> "$LOG"
if [ "$rc" -eq 124 ]; then
  echo "timeout=1 (limit ${TIMEOUT_S}s; process killed by timeout)" >> "$LOG"
else
  echo "timeout=0" >> "$LOG"
fi

snapshot after

cat > "$STATUS" <<EOF
{"model": "$MODEL", "exit_code": $rc, "timeout_s": $TIMEOUT_S,
 "duration_s": $DURATION, "end": "$(date -Is)",
 "weights": "${WEIGHTS:-default}", "calib_n": $CALIB}
EOF

echo "exit_code=$rc duration_s=$DURATION"
exit "$rc"
