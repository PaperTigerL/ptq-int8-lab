#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
CUDA_HOME=${CUDA_HOME:-$HOME/cuda}
LIBTORCH=${LIBTORCH:-$HOME/libtorch}
g++ -O3 -std=c++17 probe_overhead.cpp build/int8_kernels.o -o build/probe_overhead -I. \
    -I$LIBTORCH/include -I$LIBTORCH/include/torch/csrc/api/include -I$CUDA_HOME/include \
    -L$LIBTORCH/lib -Wl,--no-as-needed -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda \
    -L$CUDA_HOME/lib64 -lcublas -lcudart -Wl,-rpath,$LIBTORCH/lib -Wl,-rpath,$CUDA_HOME/lib64 -Wl,-rpath,/usr/lib/wsl/lib
echo build: build/probe_overhead
