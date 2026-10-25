#!/usr/bin/env bash
# Stage 0 运行器：固定线程、记录 CPU/GPU 前后快照、退出码与超时。
# 用法：_stage0_run.sh <model> <batch-sizes> [weights] [warmup] [iters]
# 环境：TIMEOUT_S（默认 1200 秒）
set -u
cd /mnt/d/WorkBuddy/ShowBro/lab/09-ptq-toolkit
mkdir -p results/stage0
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=""

SMI=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)

MODEL="${1:-resnet18}"
BATCHES="${2:-1,8,32}"
WEIGHTS="${3:-}"
WARMUP="${4:-30}"
ITERS="${5:-500}"
TIMEOUT_S="${TIMEOUT_S:-1200}"
TAG="${TAG:-}"
PIN_CPUS="${PIN_CPUS:-}"
SETTLE_S="${SETTLE_S:-0}"
OUT_JSON="${OUT_JSON:-}"

LOG="results/stage0/stage0_${MODEL}${TAG}.log"
STATUS="results/stage0/stage0_${MODEL}${TAG}.status"

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
  } > "results/stage0/nvidia_smi_${tag}_${MODEL}${TAG}.txt" 2>&1
}

: > "$LOG"
echo "start=$(date -Is) pid=$$ timeout_s=$TIMEOUT_S" >> "$LOG"
echo "model=$MODEL batches=$BATCHES weights=${WEIGHTS:-(none)} \
warmup=$WARMUP iters=$ITERS pin=${PIN_CPUS:-(none)} settle=${SETTLE_S}s" >> "$LOG"
snapshot before

if [ "$SETTLE_S" -gt 0 ]; then
  sleep "$SETTLE_S"
fi

EXTRA=()
if [ -n "$WEIGHTS" ]; then EXTRA=(--weights "$WEIGHTS"); fi
if [ -n "$OUT_JSON" ]; then EXTRA+=(--out "$OUT_JSON"); fi
PREFIX=()
if [ -n "$PIN_CPUS" ]; then PREFIX=(taskset -c "$PIN_CPUS"); fi

START_EPOCH=$(date +%s)
stdbuf -oL -eL timeout --signal=TERM "$TIMEOUT_S" \
  "${PREFIX[@]}" /home/user/pyenv-torch/bin/python -u tools/stage0_bench.py \
  --model "$MODEL" --batch-sizes "$BATCHES" --threads 4 \
  --warmup "$WARMUP" --iters "$ITERS" --runs 3 "${EXTRA[@]}" \
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
 "duration_s": $DURATION, "start": "$(date -Is)",
 "weights": "${WEIGHTS:-none}", "iters": $ITERS, "warmup": $WARMUP}
EOF

echo "exit_code=$rc duration_s=$DURATION"
exit "$rc"
