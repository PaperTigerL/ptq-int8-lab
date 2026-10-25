#!/usr/bin/env bash
# H-20261023-021 最终验收：前 2000 张拆成 4 个 500 张 chunk，每 chunk 独立进程 + no-reset。
# 每个进程内 arena 单调递增、物理上不复用地址；进程之间互不影响。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy
BIN="$HERE/ncnn_cuda_eval_h015_ptq"
P="$LAB/models/resnet18_cifar10_int8_old.param"
B="$LAB/models/resnet18_cifar10_int8_old.bin"
OUT="$HERE/results-h021"
PASSES="${1:-2}"
mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "FAIL missing $BIN"; exit 1; }
SMI="$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)"

{
  echo "== nvidia-smi before =="
  "$SMI" || true
  echo "== compute apps =="
  "$SMI" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
} > "$OUT/nvidia_smi_before_chunks.txt" 2>&1

for pass in $(seq 1 "$PASSES"); do
  for path in default fallback; do
    for batch in 64 256; do
      for k in 0 1 2 3; do
        tag="${path}_b${batch}_c${k}_p${pass}"
        log="$OUT/${tag}.log"
        if [ "$path" = "default" ]; then
          env -u NCNN_CUDA_IMPLICIT "$BIN" "$P" "$B" "$LAB/data_raw/h021/chunk${k}_500.raw" \
            --int8 --limit 500 --batch "$batch" --repeat 1 --warmup 20 --no-reset > "$log" 2>&1
        else
          NCNN_CUDA_IMPLICIT=0 "$BIN" "$P" "$B" "$LAB/data_raw/h021/chunk${k}_500.raw" \
            --int8 --limit 500 --batch "$batch" --repeat 1 --warmup 20 --no-reset > "$log" 2>&1
        fi
        rc=$?
        echo "exit_code=$rc" >> "$log"
        acc=$(awk '/^精度 [0-9]+\.[0-9]+%/{v=$2; gsub(/%/,"",v); a=v} END{print a}' "$log")
        cnt=$(awk '/^精度 [0-9]+\.[0-9]+%/{gsub(/[()]/,"",$3); split($3,x,"/"); a=x[1]} END{print a}' "$log")
        ms=$(awk '/^最优 GPU kernel/{print $4}' "$log")
        wall=$(awk '/^最优 含传输墙钟/{print $4}' "$log")
        echo "$tag rc=$rc acc=$acc correct=$cnt gpu_ms=$ms wall_ms=$wall"
      done
    done
  done
done

{
  echo "== nvidia-smi after =="
  "$SMI" || true
  echo "== compute apps =="
  "$SMI" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
} > "$OUT/nvidia_smi_after_chunks.txt" 2>&1
