#!/usr/bin/env bash
# 只编 implicit_conv.cu（迭代时用；build.sh 会全量编）。
set -e
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/libtorch/lib:$HOME/cuda/lib64
nice -n 19 $HOME/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_75 -c implicit_conv.cu -o build/implicit_conv.o
echo "OK implicit_conv.o"
