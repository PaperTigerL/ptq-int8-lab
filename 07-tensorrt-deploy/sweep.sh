#!/usr/bin/env bash
# 全配置复测，结果落盘（不要 tail 截断，不然前面的配置全丢）。
#
# 两个坑（都踩过）：
#   1) WSL 命令经 Git Bash 转发时会吃掉 $VAR —— 必须写成脚本文件
#   2) 这台机器常年有别进程占 GPU，同一份 engine 空闲 0.0138 / 被占 0.76 ms/img
#      —— 所以多轮取最小，并把每轮 GPU 占用打出来供追溯
#
# 用法: bash sweep.sh [每轮数] [输出文件]
set -u
cd /mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy/scripts
export TRT_WORKDIR=/mnt/d/WorkBuddy/_scratch/libtorch_cuda
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
P=$HOME/pyenv-torch/bin/python
REPEAT=${1:-10}
OUT=${2:-/mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy/results/sweep.txt}
mkdir -p "$(dirname "$OUT")"

{
    echo "# sweep $(date '+%F %T')  repeat=$REPEAT"
    echo "# GPU 常态被别的进程占用，这里报的是多轮最小值"
} > "$OUT"

run() {
    local model=$1 batch=$2 skip=$3
    echo "--- $model batch $batch skip $skip (repeat $REPEAT)" | tee -a "$OUT"
    $P trt_eval.py --model "$model" --batch "$batch" --skip "$skip" \
        --repeat "$REPEAT" 2>&1 | tee -a "$OUT" | grep -E "复用|built|最优|精度 "
}

run fp16 512   0
run fp16 1024  0
run fp16 2048  0
run fp16 3072  0
run fp16 4096  0
run fp16 5120  0
run fp32 512   0
run int8 512   1000
run int8 1024  1000
run int8 2048  1000

echo "结果已写入 $OUT"
