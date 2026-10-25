#!/usr/bin/env bash
# 构建二分定位工具。
set -e
cd "$(dirname "$0")"
CUDA_HOME=${CUDA_HOME:-$HOME/cuda}
LIBTORCH=${LIBTORCH:-$HOME/libtorch}
CUBLAS_DIR=$CUDA_HOME/lib64
mkdir -p build
bash build.sh > /dev/null
g++ -O2 -std=c++17 locate_block.cpp build/int8_kernels.o build/implicit_conv.o build/cifar10.o \
    -o build/locate_block -I. \
    -I$LIBTORCH/include -I$LIBTORCH/include/torch/csrc/api/include -I$CUDA_HOME/include \
    -L$LIBTORCH/lib -Wl,--no-as-needed -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda \
    -L$CUBLAS_DIR -lcublas -lcudart \
    -Wl,-rpath,$LIBTORCH/lib -Wl,-rpath,$CUBLAS_DIR
echo "构建完成: build/locate_block"
