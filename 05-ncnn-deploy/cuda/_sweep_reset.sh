#!/usr/bin/env bash
# 扫 reset_every：找到「精度正确」和「显存够用」的交点。
# 注意：WSL 命令经 Git Bash 转发会吃掉 $VAR，循环变量必须写在脚本文件里。
set -u
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64:${LD_LIBRARY_PATH:-}

LAB=/mnt/d/WorkBuddy/ShowBro/lab
BIN=$LAB/05-ncnn-deploy/cuda/ncnn_cuda_eval
MODELS=$LAB/05-ncnn-deploy/models
DATA=$LAB/05-ncnn-deploy/data_raw/test_batch.raw

for R in 1 2 4 8 32; do
    echo "===== reset-every $R ====="
    "$BIN" "$MODELS/resnet18_cifar10.param" "$MODELS/resnet18_cifar10.bin" "$DATA" \
        --fp32 --limit 200 --batch 64 --repeat 1 --warmup 5 --reset-every "$R" 2>&1 \
        | grep -E "round 1|精度"
done

echo "===== no-reset（精度基准）====="
"$BIN" "$MODELS/resnet18_cifar10.param" "$MODELS/resnet18_cifar10.bin" "$DATA" \
    --fp32 --limit 200 --batch 64 --repeat 1 --warmup 5 --no-reset 2>&1 \
    | grep -E "round 1|精度"
