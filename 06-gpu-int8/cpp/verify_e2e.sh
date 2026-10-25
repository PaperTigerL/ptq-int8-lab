#!/usr/bin/env bash
# 端到端验证：GPU int8 引擎 vs numpy 独立参考实现。
#
# ⭐ 这是**唯一的主判据**（相对早期的逐层 probe）：
#   中间张量（im2col 布局、int32 中间值）的排布是**实现细节**，
#   不同实现可以有不同排布但最终输出完全一致。
#   只有最终输出（logits + 精度）能证明语义正确。
#
# 三步：
#   1. numpy 独立实现算一遍完整前向，导出参考 logits
#   2. GPU 引擎算一遍，导出 logits
#   3. 比对：max|diff| / argmax 一致率 / 双方精度
#
# 判据（实测校准）：
#   精度差 ≤ 1.0 pp     → 通过（PTQ 量化噪声量级）
#   argmax 一致率 ≥ 90% → 通过
set -e
cd "$(dirname "$0")"

CUDA_HOME=${CUDA_HOME:-$HOME/cuda}
LIBTORCH=${LIBTORCH:-$HOME/libtorch}
CUBLAS_DIR=$CUDA_HOME/lib64
DATA=${DATA:-/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw}
N=${N:-1000}
BATCH=${BATCH:-512}
OUT=../results/cmp
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LIBTORCH/lib:$CUBLAS_DIR

mkdir -p "$OUT"
bash build.sh > /dev/null

echo "════ 1/3  numpy 参考实现（${N} 张，纯 numpy 逐层算）════"
source $HOME/pyenv-torch/bin/activate
python ref_e2e.py --limit "$N" --chunk 500 --w-bits 8 --a-bits 8 \
    --out "$OUT/ref${N}.bin" | tail -2

echo ""
echo "════ 2/3  GPU 引擎（${N} 张，batch=${BATCH}）════"
./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" \
    --limit "$N" --batch "$BATCH" --w-bits 8 --a-bits 8 \
    --dump "$OUT/gpu${N}.bin" | grep -E "精度|稳态|加速|导出"

echo ""
echo "════ 3/3  比对 ════"
python cmp_e2e.py --gpu "$OUT/gpu${N}.bin" --ref "$OUT/ref${N}.bin" || true

# ---- 可选：二分定位分歧起点（只有当端到端不一致时才需要）----
# 端到端比对只说「不一致」，不说是哪一层。二分定位跑一遍 8 个 block
# 的中间激活，直接看出分歧从哪开始 —— 比逐层编 probe 快一个数量级。
if [ "${LOCATE:-0}" = "1" ]; then
    echo ""
    echo "════ 二分定位分歧起点 ════"
    bash build_locate.sh > /dev/null
    LN=${LOCATE_N:-20}
    ./build/locate_block resnet18_folded.bin calib_minmax.bin "$DATA" "$LN"
    mkdir -p "$OUT/mid"
    cp /tmp/gpu_mid_b*.bin /tmp/gpu_mid_stem.bin "$OUT/mid/" 2>/dev/null || true
    python ref_e2e.py --limit "$LN" --chunk "$LN" --out /dev/null \
        --dump-mid "$OUT/mid/ref_mid_b" | tail -8
    python locate_block.py --n "$LN" --n-blocks 8 \
        --gpu-prefix "$OUT/mid/gpu_mid_b" --ref-prefix "$OUT/mid/ref_mid_b"
fi
