#!/usr/bin/env bash
# 用 fork 自带的 ncnn2int8 生成「老格式」int8 模型。
#
# 为什么要重新生成：老 fork 的 ModelBin 不认识新版 ncnn2int8 写进去的 weight flag_struct
# （误解析成 index_array/quantization_value → "ModelBin read index_array failed" → segfault）。
# fp32 部分格式兼容，所以 fp32 模型能跑通，int8 不能。
#
# 为什么手动编而不是 make：tools/quantize/CMakeLists.txt 把整个目录包在
# if(OpenCV_FOUND) 里，而这台机器没装 OpenCV。但 ncnn2int8 本身并不需要 OpenCV
# （需要 OpenCV 的是 ncnn2table，用来读图片做校准；我们已有现成的 table）。
set -eu
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64:${LD_LIBRARY_PATH:-}

NCNN=/home/user/ncnn-cuda
CUDA=/home/user/cuda
LAB=/mnt/d/WorkBuddy/ShowBro/lab
MODELS=$LAB/05-ncnn-deploy/models
TABLES=$LAB/05-ncnn-deploy/tables

echo "=== 编译 ncnn2int8 ==="
g++ -O2 -std=c++11 -fopenmp \
    "$NCNN/tools/quantize/ncnn2int8.cpp" \
    -o "$NCNN/build/tools/ncnn2int8" \
    -I"$NCNN/src" -I"$NCNN/build/src" -I"$CUDA/include" \
    -L"$NCNN/build/src" -L"$CUDA/lib64" \
    -Wl,--start-group -lncnn -lncnn_cuda_lib -Wl,--end-group \
    -lcudart -lcublas -lpthread -ldl \
    -Wl,-rpath,"$NCNN/build/src" 2>&1 | tail -10
ls -la "$NCNN/build/tools/ncnn2int8"

echo "=== 转换（kl table）==="
"$NCNN/build/tools/ncnn2int8" \
    "$MODELS/resnet18_cifar10.param" \
    "$MODELS/resnet18_cifar10.bin" \
    "$MODELS/resnet18_cifar10_int8_old.param" \
    "$MODELS/resnet18_cifar10_int8_old.bin" \
    "$TABLES/resnet18_cifar10_kl.table" 2>&1 | tail -10

echo "=== 产物 ==="
ls -la "$MODELS"/resnet18_cifar10_int8_old.* 2>/dev/null
