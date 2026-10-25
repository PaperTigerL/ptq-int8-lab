#!/usr/bin/env bash
# 单层数值校验的构建脚本（与 build.sh 同一套两段编译规则）。
set -e
cd "$(dirname "$0")"

CUDA_HOME=${CUDA_HOME:-$HOME/cuda}
LIBTORCH=${LIBTORCH:-$HOME/libtorch}
CUBLAS_DIR=$CUDA_HOME/lib64
mkdir -p build

# 增量：.o 比源新就跳过（verify.sh 会先跑 build.sh，这里再全量编一遍纯属浪费
# 4 分钟；而写错时 nvcc 的报错仍然会在这个脚本里出现）。
for pair in "int8_kernels.cu build/int8_kernels.o" "implicit_conv.cu build/implicit_conv.o"; do
    set -- $pair
    if [ ! -f "$2" ] || [ "$1" -nt "$2" ]; then
        $CUDA_HOME/bin/nvcc -O3 -std=c++17 -arch=sm_75 -c "$1" -o "$2"
    fi
done

g++ -O2 -std=c++17 \
    check_layer.cpp build/int8_kernels.o build/implicit_conv.o -o build/check_layer \
    -I. \
    -I$LIBTORCH/include \
    -I$LIBTORCH/include/torch/csrc/api/include \
    -I$CUDA_HOME/include \
    -L$LIBTORCH/lib \
    -Wl,--no-as-needed \
    -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda \
    -L$CUBLAS_DIR -lcublas -lcudart \
    -Wl,-rpath,$LIBTORCH/lib -Wl,-rpath,$CUBLAS_DIR

echo "构建完成: build/check_layer"
echo "用法: LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LIBTORCH/lib:$CUBLAS_DIR ./build/check_layer <model> <calib> <layer_idx> [--n N]"
