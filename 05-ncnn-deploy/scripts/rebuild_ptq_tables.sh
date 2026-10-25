#!/usr/bin/env bash
# 用 ptq-lab 的三份 W8A8 配置重建 ncnn 量化表，并端到端测精度。
#
# 每一步都是「可单独重跑」的：表在 tables/，模型在 models/_symfix/。
# 换策略/换口径后跑这一条就能拿到完整的三策略对照。
#
# 用法（WSL，在 05-ncnn-deploy/ 下）：
#   ./scripts/rebuild_ptq_tables.sh                 # 三个策略全跑
#   ./scripts/rebuild_ptq_tables.sh minmax entropy  # 只跑指定策略
#   NO_SYM_FIX=1 ./scripts/rebuild_ptq_tables.sh    # 复现 2026-10-18 修正前的旧行为
set -euo pipefail

LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LAB"

PTQ_SCALES="$(cd "$LAB/../03-ptq-lab" && pwd)/results/cifar10_resnet18"
PARAM="models/resnet18_cifar10_opt.param"
BIN="models/resnet18_cifar10_opt.bin"
BASELINE="tables/resnet18_cifar10_kl.table"
INT8="${NCNN2INT8:-$HOME/ncnn-build/ncnn/build/tools/quantize/ncnn2int8}"
RUNNER="cpp/build/cifar10_eval"
WORK="models/_symfix"
THREADS="${NCNN_THREADS:-1}"

# 码空间换算：默认开（127/255）。NO_SYM_FIX=1 关掉，用于复现旧行为。
SYM_FLAG=""
[ "${NO_SYM_FIX:-0}" = "1" ] && SYM_FLAG="--no-sym-fix"

STRATEGIES="${*:-minmax percentile entropy}"

for f in "$PARAM" "$BIN" "$BASELINE" "$INT8" "$RUNNER"; do
    [ -f "$f" ] || { echo "[缺] $f"; exit 1; }
done

mkdir -p "$WORK"

for name in $STRATEGIES; do
    scales="$PTQ_SCALES/scales_w8a8_${name}_mine.json"
    table="tables/resnet18_cifar10_ptq_${name}_mine.table"
    [ -f "$scales" ] || { echo "[缺] $scales，跳过 $name"; continue; }

    echo
    echo "==================== $name ===================="
    python3 scripts/make_table.py --scales "$scales" --param "$PARAM" \
        --baseline "$BASELINE" --out "$table" $SYM_FLAG 2>&1 | tail -2

    "$INT8" "$PARAM" "$BIN" "$WORK/$name.param" "$WORK/$name.bin" "$table" \
        > /dev/null 2>&1

    "./$RUNNER" --param "$WORK/$name.param" --bin "$WORK/$name.bin" \
        --data data_raw/test_batch.raw --threads "$THREADS" --tag "$name" \
        2>&1 | grep -E "精度|ms/张"
done

echo
echo "表在 tables/，模型在 $WORK/。对照 Python 侧数字见 docs/阶段2 §3.6。"
