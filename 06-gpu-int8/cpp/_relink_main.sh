#!/usr/bin/env bash
# 只重编主机侧并链接（.cu 未改时用；A/B 前确保跑的是当前 main.cpp）。
set -e
cd "$(dirname "$0")"
CUDA_HOME=${CUDA_HOME:-$HOME/cuda}
LIBTORCH=${LIBTORCH:-$HOME/libtorch}
g++ -O3 -std=c++17 -c main.cpp -o build/main.o -I. \
    -I$LIBTORCH/include -I$LIBTORCH/include/torch/csrc/api/include -I$CUDA_HOME/include
g++ build/main.o build/int8_kernels.o build/implicit_conv.o build/cifar10.o -o build/gpu_int8 \
    -L$LIBTORCH/lib -Wl,--no-as-needed -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda \
    -L$CUDA_HOME/lib64 -lcublas -lcudart -Wl,-rpath,$LIBTORCH/lib -Wl,-rpath,$CUDA_HOME/lib64 -Wl,-rpath,/usr/lib/wsl/lib
echo relink: build/gpu_int8
