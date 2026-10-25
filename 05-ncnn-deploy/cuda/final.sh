#!/usr/bin/env bash
# ncnn + CUDA 最终实测。
# 注意：WSL 命令经 Git Bash 转发会吃掉 $VAR，所以必须写成脚本文件。
set -u
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64:${LD_LIBRARY_PATH:-}

LAB=/mnt/d/WorkBuddy/ShowBro/lab
BIN=$LAB/05-ncnn-deploy/cuda/ncnn_cuda_eval
MODELS=$LAB/05-ncnn-deploy/models
DATA=$LAB/05-ncnn-deploy/data_raw/test_batch.raw
OUT=$LAB/05-ncnn-deploy/cuda/results

mkdir -p "$OUT"

# 样本数只能用 200：唯一精度正确的配置（不复用显存）每图要吃掉 ~1.3 MB 显存且
# 从不释放，1000 张就把 256 MB 的 arena 撑爆（实测报 arena 溢出）。
# 200 张是「精度正确」这个前提下能跑的最大规模。
N=200

echo "########## A. fp32 / 不复用显存（唯一精度正确的配置）/ $N 张 ##########"
"$BIN" "$MODELS/resnet18_cifar10.param" "$MODELS/resnet18_cifar10.bin" "$DATA" \
    --fp32 --limit $N --batch 64 --repeat 3 --warmup 5 --no-reset 2>&1 | tee "$OUT/cuda_fp32_noreset.txt" | tail -12

echo
echo "########## B. fp32 / 每张复用显存（最快，但精度不可用）/ $N 张 ##########"
"$BIN" "$MODELS/resnet18_cifar10.param" "$MODELS/resnet18_cifar10.bin" "$DATA" \
    --fp32 --limit $N --batch 64 --repeat 3 --warmup 5 --reset-every 1 2>&1 | tee "$OUT/cuda_fp32_reuse.txt" | tail -12

echo
echo "########## C. int8 / 不复用显存 / $N 张 ##########"
"$BIN" "$MODELS/resnet18_cifar10_int8_old.param" "$MODELS/resnet18_cifar10_int8_old.bin" "$DATA" \
    --int8 --limit $N --batch 64 --repeat 3 --warmup 5 --no-reset 2>&1 | tee "$OUT/cuda_int8_noreset.txt" | tail -12
