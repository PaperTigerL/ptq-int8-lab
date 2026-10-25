#!/usr/bin/env bash
# 看每个 kernel 变体的寄存器占用与本地内存溢出（spill）。
set -u
cd "$(dirname "$0")"
mkdir -p build
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/libtorch/lib:$HOME/cuda/lib64
$HOME/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 -Xptxas -v \
    -c implicit_conv.cu -o build/imp_probe.o > build/ptxas.log 2>&1
echo "done rc=$?"
