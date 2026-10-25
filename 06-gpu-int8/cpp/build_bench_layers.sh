#!/usr/bin/env bash
# 构建逐层 roofline 基准（复用 build.sh 已经编好的 int8_kernels.o）。
set -e
cd "$(dirname "$0")"
CUDA_HOME=${CUDA_HOME:-$HOME/cuda}
LIBTORCH=${LIBTORCH:-$HOME/libtorch}
g++ -O3 -std=c++17 bench_layers.cpp build/int8_kernels.o build/implicit_conv.o -o build/bench_layers -I. \
    -I$LIBTORCH/include -I$LIBTORCH/include/torch/csrc/api/include -I$CUDA_HOME/include \
    -L$LIBTORCH/lib -Wl,--no-as-needed -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda \
    -L$CUDA_HOME/lib64 -lcublas -lcudart \
    -Wl,-rpath,$LIBTORCH/lib -Wl,-rpath,$CUDA_HOME/lib64 -Wl,-rpath,/usr/lib/wsl/lib
echo "build: build/bench_layers"
