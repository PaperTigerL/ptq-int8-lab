#!/usr/bin/env bash
# 跑 ncnn + CUDA 评测。
#
# 注意：WSL 命令经 Git Bash 转发会吃掉 $VAR，所以必须写成脚本文件。
#      并且要用 `wsl.exe bash -lc 'bash <脚本>'` 调用，不能直接把 WSL 路径
#      当 Windows 路径传给 wsl.exe（会被 Git Bash 拼成 C:/.../mnt/d/...）。
set -u
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64:${LD_LIBRARY_PATH:-}

LAB=/mnt/d/WorkBuddy/ShowBro/lab
BIN=$LAB/05-ncnn-deploy/cuda/ncnn_cuda_eval
MODELS=$LAB/05-ncnn-deploy/models
DATA=$LAB/05-ncnn-deploy/data_raw/test_batch.raw

MODE=${1:-int8}      # int8 | fp32
LIMIT=${2:-10000}    # 样本数
BATCH=${3:-256}      # 流水线一次提交多少张
REPEAT=${4:-3}

case "$MODE" in
    int8) PARAM=$MODELS/resnet18_cifar10_int8_old.param; MODEL=$MODELS/resnet18_cifar10_int8_old.bin; FLAG=--int8 ;;
    fp32) PARAM=$MODELS/resnet18_cifar10.param;           MODEL=$MODELS/resnet18_cifar10.bin;           FLAG=--fp32 ;;
    *) echo "usage: $0 [int8|fp32] [limit] [batch] [repeat]"; exit 2 ;;
esac

"$BIN" "$PARAM" "$MODEL" "$DATA" $FLAG \
       --limit "$LIMIT" --batch "$BATCH" --repeat "$REPEAT" --warmup 20
