#!/usr/bin/env bash
# 构建按层选择表生成器（复用 build.sh 已编好的 int8_kernels.o / implicit_conv.o）。
set -e
cd "$(dirname "$0")"
CUDA_HOME=${CUDA_HOME:-$HOME/cuda}
LIBTORCH=${LIBTORCH:-$HOME/libtorch}
g++ -O3 -std=c++17 select_layers.cpp build/int8_kernels.o build/implicit_conv.o -o build/select_layers -I. \
    -I$LIBTORCH/include -I$LIBTORCH/include/torch/csrc/api/include -I$CUDA_HOME/include \
    -L$LIBTORCH/lib -Wl,--no-as-needed -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda \
    -L$CUDA_HOME/lib64 -lcublas -lcudart \
    -Wl,-rpath,$LIBTORCH/lib -Wl,-rpath,$CUDA_HOME/lib64 -Wl,-rpath,/usr/lib/wsl/lib
echo "build: build/select_layers"
