#!/usr/bin/env bash
# 用 LOG_LAYERS 版本跑 1 张，看每层到底走没走 CUDA。
# 判据：日志里 forward 行带 cuda 的走 GPU，不带的就是掉回 CPU（会引发 ping-pong）。
#
# ⚠️ 历史教训：这份脚本早期跑出来的「0 个 cuda 层」曾经把我带偏，以为是 fork 的
#    层调度坏了。真正的原因是 ncnn_cuda_eval 当时往 Extractor 里塞的是 ncnn::Mat，
#    走的是 CPU 通路 —— 跟层调度一点关系都没有（详见 README 第二节）。
#    另外：LOG_LAYERS 一旦打开就会被 CMakeLists.txt 拼进全局 CMAKE_CXX_FLAGS，
#    整个库每层 forward 都 fprintf 一行，测出来的延迟完全不可用。
#    用完务必跑 _rebuild.sh（带 -DLOG_LAYERS=OFF）恢复。
set -u
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64:${LD_LIBRARY_PATH:-}

LAB=/mnt/d/WorkBuddy/ShowBro/lab
BIN=$LAB/05-ncnn-deploy/cuda/ncnn_cuda_eval_log
MODELS=$LAB/05-ncnn-deploy/models
DATA=$LAB/05-ncnn-deploy/data_raw/test_batch.raw
LOG=$LAB/05-ncnn-deploy/cuda/results_layers.log
mkdir -p "$(dirname "$LOG")"

"$BIN" "$MODELS/resnet18_cifar10_int8_old.param" \
       "$MODELS/resnet18_cifar10_int8_old.bin" \
       "$DATA" --int8 --limit 1 --warmup 0 --repeat 1 > "$LOG" 2>&1

echo "=== 总行数 ==="
wc -l "$LOG"
echo "=== 前 15 行 ==="
head -15 "$LOG"
echo "=== forward 行里不带 cuda 的（掉回 CPU 的层）==="
grep -i "forward" "$LOG" | grep -vi "cuda" | head -30
echo "=== forward 行里带 cuda 的数量 ==="
grep -i "forward" "$LOG" | grep -i "cuda" | wc -l
