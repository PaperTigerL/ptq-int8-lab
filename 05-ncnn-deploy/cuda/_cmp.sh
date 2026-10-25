#!/usr/bin/env bash
# 编译并运行逐层对比工具（CPU vs GPU）。
# 注意：WSL 路径必须走 `wsl.exe bash -lc 'bash <脚本>'`，不能当 Windows 路径传。
set -eu
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64:${LD_LIBRARY_PATH:-}

NCNN=/home/user/ncnn-cuda
CUDA=/home/user/cuda
LAB=/mnt/d/WorkBuddy/ShowBro/lab
HERE="$LAB/05-ncnn-deploy/cuda"

g++ -O2 -std=c++11 -fopenmp \
    "$HERE/ncnn_cuda_cmp.cpp" \
    "$LAB/04-libtorch-cuda/cpp/cifar10.cpp" \
    -o "$HERE/ncnn_cuda_cmp" \
    -I"$LAB/04-libtorch-cuda/cpp" \
    -I"$NCNN/src" -I"$NCNN/build/src" -I"$CUDA/include" \
    -L"$NCNN/build/src" -L"$CUDA/lib64" \
    -Wl,--start-group -lncnn -lncnn_cuda_lib -Wl,--end-group \
    -lcudart -lcublas -lpthread -ldl \
    -Wl,-rpath,"$NCNN/build/src" -Wl,-rpath,"$CUDA/lib64" 2>&1 | tail -10

MODELS=$LAB/05-ncnn-deploy/models
DATA=$LAB/05-ncnn-deploy/data_raw/test_batch.raw
MODE=${1:-fp32}
IMG=${2:-0}

if [ "$MODE" = "int8" ]; then
    PARAM=$MODELS/resnet18_cifar10_int8_old.param; MODEL=$MODELS/resnet18_cifar10_int8_old.bin; FLAG=--int8
else
    PARAM=$MODELS/resnet18_cifar10.param; MODEL=$MODELS/resnet18_cifar10.bin; FLAG=--fp32
fi

"$HERE/ncnn_cuda_cmp" "$PARAM" "$MODEL" "$DATA" $FLAG --img "$IMG"
