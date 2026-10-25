#!/usr/bin/env bash
# H-20261023-021 验收：default implicit / NCNN_CUDA_IMPLICIT=0 fallback × batch 64/256。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
BIN="$HERE/ncnn_cuda_eval_final"
P="$LAB/models/resnet18_cifar10_int8_old.param"
B="$LAB/models/resnet18_cifar10_int8_old.bin"
DATA="$LAB/data_raw/test_batch.raw"
OUT="$HERE/results-h021"
MODE="${1:-full}"
mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "FAIL missing $BIN"; exit 1; }
SMI="$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)"
LIMIT=2000; REPEAT=3; WARMUP=20
if [ "$MODE" = "smoke" ]; then LIMIT=100; REPEAT=1; WARMUP=5; fi

{
  echo "== nvidia-smi before =="
  "$SMI" || true
  echo "== compute apps =="
  "$SMI" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
} > "$OUT/nvidia_smi_before_${MODE}.txt" 2>&1

run_one() {
  local path="$1" batch="$2" tag="$3"
  local log="$OUT/${tag}.log"
  local rc
  if [ "$path" = "default" ]; then
    env -u NCNN_CUDA_IMPLICIT "$BIN" "$P" "$B" "$DATA" --int8 \
      --limit "$LIMIT" --batch "$batch" --repeat "$REPEAT" --warmup "$WARMUP" \
      --reset-every 1 > "$log" 2>&1
  else
    NCNN_CUDA_IMPLICIT=0 "$BIN" "$P" "$B" "$DATA" --int8 \
      --limit "$LIMIT" --batch "$batch" --repeat "$REPEAT" --warmup "$WARMUP" \
      --reset-every 1 > "$log" 2>&1
  fi
  rc=$?
  echo "exit_code=$rc" >> "$log"
  local acc
  acc=$(awk '/^精度 [0-9]+\.[0-9]+%/{v=$2; gsub(/%/,"",v); a=v} END{print a}' "$log")
  local ms
  ms=$(awk '/^最优 GPU kernel/{print $4}' "$log")
  echo "$tag rc=$rc acc=$acc gpu_ms=$ms"
  return "$rc"
}

if [ "$MODE" = "smoke" ]; then
  run_one default 64 "smoke_default_b64"
  run_one fallback 64 "smoke_fallback_b64"
else
  for path in default fallback; do
    for batch in 64 256; do
      run_one "$path" "$batch" "${path}_b${batch}"
    done
  done
fi

{
  echo "== nvidia-smi after =="
  "$SMI" || true
  echo "== compute apps =="
  "$SMI" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
} > "$OUT/nvidia_smi_after_${MODE}.txt" 2>&1
