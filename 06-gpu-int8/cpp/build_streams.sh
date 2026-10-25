#!/usr/bin/env bash
# 构建多流（跨图并行）吞吐 harness。复用 build.sh 编好的 .o。
set -e
cd "$(dirname "$0")"
CUDA_HOME=${CUDA_HOME:-$HOME/cuda}
LIBTORCH=${LIBTORCH:-$HOME/libtorch}
g++ -O3 -std=c++17 bench_streams.cpp build/int8_kernels.o build/implicit_conv.o \
    build/cifar10.o \
    -o build/bench_streams -I. \
    -I$LIBTORCH/include -I$LIBTORCH/include/torch/csrc/api/include -I$CUDA_HOME/include \
    -L$LIBTORCH/lib -Wl,--no-as-needed -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda \
    -L$CUDA_HOME/lib64 -lcublas -lcudart \
    -Wl,-rpath,$LIBTORCH/lib -Wl,-rpath,$CUDA_HOME/lib64 -Wl,-rpath,/usr/lib/wsl/lib
echo "build: build/bench_streams"
