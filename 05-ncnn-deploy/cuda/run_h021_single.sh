#!/usr/bin/env bash
# H-20261023-021 单进程 2000 张验收：blob arena 单调（reset 自动 defer）+ work arena 每图回退。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
BIN="$HERE/ncnn_cuda_eval_2000c"
P="$LAB/models/resnet18_cifar10_int8_old.param"
B="$LAB/models/resnet18_cifar10_int8_old.bin"
DATA="$LAB/data_raw/test_batch.raw"
OUT="$HERE/results-h021"
mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "FAIL missing $BIN"; exit 1; }
SMI="$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)"

{
  echo "== nvidia-smi before =="
  "$SMI" || true
  echo "== compute apps =="
  "$SMI" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
} > "$OUT/nvidia_smi_before_single.txt" 2>&1

for path in default fallback; do
  for batch in 64 256; do
    tag="single_${path}_b${batch}"
    log="$OUT/${tag}.log"
    if [ "$path" = "default" ]; then
      env -u NCNN_CUDA_IMPLICIT "$BIN" "$P" "$B" "$DATA" --int8 \
        --limit 2000 --batch "$batch" --repeat 1 --warmup 20 --reset-every 1 > "$log" 2>&1
    else
      NCNN_CUDA_IMPLICIT=0 "$BIN" "$P" "$B" "$DATA" --int8 \
        --limit 2000 --batch "$batch" --repeat 1 --warmup 20 --reset-every 1 > "$log" 2>&1
    fi
    rc=$?
    echo "exit_code=$rc" >> "$log"
    acc=$(awk '/^精度 [0-9]+\.[0-9]+%/{v=$2; gsub(/%/,"",v); a=v} END{print a}' "$log")
    cnt=$(awk '/^精度 [0-9]+\.[0-9]+%/{gsub(/[()]/,"",$3); split($3,x,"/"); a=x[1]} END{print a}' "$log")
    ms=$(awk '/^最优 GPU kernel/{print $4}' "$log")
    used=$(awk -F'[= ]' '/^arena_used_mb/{print $2}' "$log")
    echo "$tag rc=$rc acc=$acc correct=$cnt gpu_ms=$ms blob_used_mb=$used"
  done
done

{
  echo "== nvidia-smi after =="
  "$SMI" || true
  echo "== compute apps =="
  "$SMI" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
} > "$OUT/nvidia_smi_after_single.txt" 2>&1
