#!/usr/bin/env bash
# 生成层选择表：先小 batch 冒烟，再 batch=512 全量实测（外部 flock 包裹）。
set -u
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/libtorch/lib:$HOME/cuda/lib64
mkdir -p ../results
echo "==== clock start ===="
nvidia-smi --query-gpu=clocks.current.graphics,memory.used --format=csv,noheader
echo "==== smoke (batch=64, warmup=3, iters=2) ===="
./build/select_layers resnet18_folded.bin calib_minmax.bin --batch 64 --warmup 3 --iters 2 || { echo "SMOKE_FAIL"; exit 1; }
echo "SMOKE_OK"
echo "==== full A/B (batch=512, warmup=100, iters=15) ===="
./build/select_layers resnet18_folded.bin calib_minmax.bin --batch 512 --warmup 100 --iters 15 \
    --json ../results/layer_impl.json --header layer_impl_default.h
RC=$?
echo "==== clock end ===="
nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader
echo "RC=$RC"
exit $RC
