#!/usr/bin/env bash
# 构建 GPU int8 引擎。
#
# ⭐ 为什么要分两段编译（nvcc + g++），而不是一把 nvcc 编完：
#   · int8_kernels.cu 是纯 CUDA（自带 kernel 定义 + launcher）→ nvcc 编
#   · main.cpp / int8_engine.cuh 要用 ATen / torch 头文件，
#     那些头里有大量 nvcc 不认识的构造（模板 + C++17 特性），
#     一把 nvcc 会在 torch 头上炸一堆无关错误。
#   所以：nvcc 编 CUDA 部分 → 产出 .o；g++ 编主机部分（带 ATen 头）；
#         最后链接。
#
# ⚠️ 两个必须记住的细节（踩过才知道，且都不报错）：
#   1. arch 必须显式给 sm_75。GTX 1650 Ti 是 Turing(7.5)，nvcc 默认按本机
#      探测可能落到 sm_52（Pascal，无 int8 tensor core）——那样 GEMM 会
#      静默走慢 4 倍的非 tensor core 路径，**精度完全正常**。
#   2. libtorch 链接必须带 `-Wl,--no-as-needed`。默认 --as-needed 会把
#      libtorch_cuda.so 剔掉（is_available 的符号经 libtorch_cpu 转发），
#      结果运行时静默退到 CPU，且**没有任何报错**。
set -e
cd "$(dirname "$0")"

CUDA_HOME=${CUDA_HOME:-$HOME/cuda}
LIBTORCH=${LIBTORCH:-$HOME/libtorch}
CUBLAS_DIR=$CUDA_HOME/lib64
mkdir -p build

echo "CUDA   : $($CUDA_HOME/bin/nvcc --version | tail -2 | head -1)"
echo "libtorch: $LIBTORCH"

# --- 1) CUDA 部分（kernel + launcher）---
$CUDA_HOME/bin/nvcc -O3 -std=c++17 -arch=sm_75 \
    -c int8_kernels.cu -o build/int8_kernels.o

# implicit GEMM（直接卷积）单独一个翻译单元：
# 它的模板实例化多、编译慢（实测 144 个实例 9 分钟），拆出来以后只改它
# 就不用重编整个 kernel 文件（改一次 1-2 分钟）。
$CUDA_HOME/bin/nvcc -O3 -std=c++17 -arch=sm_75 \
    -c implicit_conv.cu -o build/implicit_conv.o

# --- 2) 主机部分（含 ATen 头，用 g++）---
g++ -O3 -std=c++17 \
    -c main.cpp -o build/main.o \
    -I. \
    -I$LIBTORCH/include \
    -I$LIBTORCH/include/torch/csrc/api/include \
    -I$CUDA_HOME/include

g++ -O3 -std=c++17 \
    -c cifar10.cpp -o build/cifar10.o \
    -I.

# --- 3) 链接 ---
g++ build/main.o build/int8_kernels.o build/implicit_conv.o build/cifar10.o \
    -o build/gpu_int8 \
    -L$LIBTORCH/lib \
    -Wl,--no-as-needed \
    -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda \
    -L$CUBLAS_DIR -lcublas -lcudart \
    -Wl,-rpath,$LIBTORCH/lib \
    -Wl,-rpath,$CUBLAS_DIR \
    -Wl,-rpath,/usr/lib/wsl/lib

echo "构建完成: build/gpu_int8"
echo "运行: LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LIBTORCH/lib:$CUBLAS_DIR ./build/gpu_int8 ..."
