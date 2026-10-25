#!/usr/bin/env bash
# 05-ncnn-deploy 的一条龙管线：ONNX → ncnn → 校准表 → INT8 → 测精度
#
# 设计原则：**每一步都可单独重跑，且中间产物都落盘**。
# 不需要「一键跑完再看结果」—— 中间哪一步错了，从那一步重来就行，
# 不用把前面几十分钟的活再干一遍。
#
# 用法（WSL，在 05-ncnn-deploy/ 目录下）：
#   ./scripts/run_pipeline.sh           # 全流程
#   ./scripts/run_pipeline.sh export    # 只导 ONNX
#   ./scripts/run_pipeline.sh convert   # 只转 ncnn + 融合
#   ./scripts/run_pipeline.sh calib     # 只出校准表
#   ./scripts/run_pipeline.sh int8      # 只出 INT8 模型
#   ./scripts/run_pipeline.sh eval      # 只测精度
#   ./scripts/run_pipeline.sh raw       # 只导 CIFAR 裸二进制
set -euo pipefail

LAB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LAB"

# ---------------------------------------------------------------- 配置
# 环境相关的路径集中在这里，换机器只改这一段。
PY="${PY:-$HOME/pyenv-torch/bin/python}"
NCNN_BUILD="${NCNN_BUILD:-$HOME/ncnn-build/ncnn/build}"
T_ONNX="$NCNN_BUILD/tools/onnx/onnx2ncnn"
T_OPT="$NCNN_BUILD/tools/ncnnoptimize"
T_TABLE="$NCNN_BUILD/tools/quantize/ncnn2table"
T_INT8="$NCNN_BUILD/tools/quantize/ncnn2int8"

DATASET="${DATASET:-cifar10}"
ARCH="${ARCH:-resnet18}"
BASE="${ARCH}_${DATASET}"      # resnet18_cifar10
MODEL_DIR="$LAB/models"
CALIB_DIR="$LAB/calib"
TABLE_DIR="$LAB/tables"
DATA_RAW="$LAB/data_raw/test_batch.raw"

# 校准口径：必须和 03-ptq-lab 一致，否则两边的数字不可比
CALIB_SIZE="${CALIB_SIZE:-1024}"
CALIB_SEED="${CALIB_SEED:-2026}"
NCNN_THREADS="${NCNN_THREADS:-1}"   # 32x32 小图上多线程反而慢，默认单线程

hr() { echo "------------------------------------------------------------"; }

check_tools() {
    local miss=0
    for t in "$T_ONNX" "$T_OPT" "$T_TABLE" "$T_INT8"; do
        [ -x "$t" ] || { echo "[缺] $t"; miss=1; }
    done
    [ -x "$LAB/cpp/build/cifar10_eval" ] || { echo "[缺] cpp/build/cifar10_eval（先在 cpp/ 下 cmake+make）"; miss=1; }
    [ $miss -eq 1 ] && exit 1
    return 0
}

step_raw() {
    hr; echo "### 1/6  导 CIFAR-10 裸二进制（C++ runner 的数据源）"
    # 磁盘上的 cifar-10-batches-py 是 pickle，不是裸二进制，必须先转
    "$PY" scripts/dump_cifar_raw.py
}

step_export() {
    hr; echo "### 2/6  PyTorch → ONNX（opset 11，自带 onnxruntime 自检）"
    "$PY" scripts/export_onnx.py --dataset "$DATASET" --arch "$ARCH"
}

step_convert() {
    hr; echo "### 3/6  ONNX → ncnn，再 ncnnoptimize 融合"
    # onnx2ncnn 输出的 Input 层不带 shape，ncnn2table 不需要它（shape 走命令行），
    # 但推理时 ncnn 会自己从输入 Mat 推，所以不用补。
    "$T_ONNX" "$MODEL_DIR/$BASE.onnx" \
              "$MODEL_DIR/$BASE.param" "$MODEL_DIR/$BASE.bin" 2>&1 | grep -v 'pnnx\|PNNX\|^$\|pip3\|github.com' || true
    "$T_OPT" "$MODEL_DIR/$BASE.param" "$MODEL_DIR/$BASE.bin" \
             "$MODEL_DIR/${BASE}_opt.param" "$MODEL_DIR/${BASE}_opt.bin" 0 \
             > "$LAB/models/ncnnoptimize.log" 2>&1 || {
        echo "[FAIL] ncnnoptimize 失败，日志在 models/ncnnoptimize.log"; exit 1; }
    echo "融合完成，日志：models/ncnnoptimize.log"
    echo "层类型分布："
    awk 'NR>2{print "    "$1}' "$MODEL_DIR/${BASE}_opt.param" | sort | uniq -c | sort -rn
}

step_calib() {
    hr; echo "### 4/6  校准集 → npy → ncnn2table（KL）"
    "$PY" scripts/dump_calib_npy.py --num "$CALIB_SIZE"
    mkdir -p "$TABLE_DIR"
    # type=1 读 npy；shape=[w,h,c] 必须显式给，ncnn2table 不从 param 的 Input 层读 shape
    "$T_TABLE" "$MODEL_DIR/${BASE}_opt.param" "$MODEL_DIR/${BASE}_opt.bin" \
               "$CALIB_DIR/filelist.txt" "$TABLE_DIR/${BASE}_kl.table" \
               type=1 method=kl shape=[32,32,3] thread=12 2>&1 | tail -2
    echo "校准表：$TABLE_DIR/${BASE}_kl.table（$(wc -l < "$TABLE_DIR/${BASE}_kl.table") 行）"
}

step_int8() {
    hr; echo "### 5/6  ncnn2int8 → INT8 模型"
    # 不传第 5 个参数（校准表）也能跑，但那样产出的是 **fp16 权重、无激活量化** 的
    # 模型，不是 INT8。见 docs/ 里的记录。
    "$T_INT8" "$MODEL_DIR/${BASE}_opt.param" "$MODEL_DIR/${BASE}_opt.bin" \
              "$MODEL_DIR/${BASE}_int8.param" "$MODEL_DIR/${BASE}_int8.bin" \
              "$TABLE_DIR/${BASE}_kl.table" 2>&1 | tail -1
    local s1 s2
    s1=$(stat -c%s "$MODEL_DIR/${BASE}_opt.bin")
    s2=$(stat -c%s "$MODEL_DIR/${BASE}_int8.bin")
    echo "FP32 bin : $((s1/1024)) KB"
    echo "INT8 bin : $((s2/1024)) KB  （$(awk "BEGIN{printf \"%.2f\", $s1/$s2}")x 压缩）"
}

step_eval() {
    hr; echo "### 6/6  测精度（单线程，全量 10000）"
    local runner="$LAB/cpp/build/cifar10_eval"
    for m in "${BASE}_opt:FP32" "${BASE}_int8:INT8"; do
        local f="${m%%:*}" name="${m##*:}"
        [ -f "$MODEL_DIR/$f.param" ] || { echo "[缺] $f，跳过"; continue; }
        echo
        "$runner" --param "$MODEL_DIR/$f.param" --bin "$MODEL_DIR/$f.bin" \
                  --data "$DATA_RAW" --threads "$NCNN_THREADS" --tag "$name"
    done
}

check_tools

case "${1:-all}" in
    raw)     step_raw ;;
    export)  step_export ;;
    convert) step_convert ;;
    calib)   step_calib ;;
    int8)    step_int8 ;;
    eval)    step_eval ;;
    all)
        step_raw; step_export; step_convert; step_calib; step_int8; step_eval
        hr; echo "全部完成。对照 03-ptq-lab 的数字在 docs/阶段1-ncnn-INT8基线.md"
        ;;
    *) echo "未知步骤：$1（可选 raw/export/convert/calib/int8/eval/all）"; exit 2 ;;
esac
