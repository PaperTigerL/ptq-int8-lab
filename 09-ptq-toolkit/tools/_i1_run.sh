#!/usr/bin/env bash
# I1 运行器：int8 conv/gemm CPU 参考实现单测，CPU-only，记录 GPU 前后快照。
# 用法：bash tools/_i1_run.sh [日志后缀]
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SMI=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)
SUFFIX="${1:-run}"
LOG="results/i1/test_int8_ref_${SUFFIX}.log"

mkdir -p results/i1
snapshot() {
  local tag="$1"
  {
    echo "== $tag @ $(date -Is) =="
    "$SMI" || true
    echo "== $tag compute apps =="
    "$SMI" --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader || true
  } > "results/i1/nvidia_smi_${tag}.txt" 2>&1
}

snapshot before
START=$(date +%s)
/home/user/pyenv-torch/bin/python -u tests/test_int8_ref.py > "$LOG" 2>&1
rc=$?
DURATION=$(( $(date +%s) - START ))
snapshot after

echo "log=$PWD/$LOG" >> "$LOG"
echo "exit_code=$rc duration_s=$DURATION" >> "$LOG"
echo "exit_code=$rc duration_s=$DURATION"
exit "$rc"
