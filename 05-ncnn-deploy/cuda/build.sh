#!/usr/bin/env bash
# 编译 ncnn + CUDA 评测器。
# 注意：WSL 命令经 Git Bash 转发会吃掉 $VAR，所以必须写成脚本文件。
set -eu

NCNN="${NCNN_DIR:-/home/user/ncnn-cuda}"
NCNN_BUILD="${NCNN_BUILD_DIR:-$NCNN/build}"
CUDA=/home/user/cuda
LAB=/mnt/d/WorkBuddy/ShowBro/lab
HERE="$LAB/05-ncnn-deploy/cuda"

export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64:${LD_LIBRARY_PATH:-}

g++ -O2 -std=c++11 -fopenmp \
    "${EVAL_SRC:-$HERE/ncnn_cuda_eval.cpp}" \
    "$LAB/04-libtorch-cuda/cpp/cifar10.cpp" \
    -o "${EVAL_BIN:-$HERE/ncnn_cuda_eval}" \
    -I"$HERE" \
    -I"$LAB/04-libtorch-cuda/cpp" \
    -I"$NCNN/src" \
    -I"$NCNN_BUILD/src" \
    -I"$CUDA/include" \
    -L"$NCNN_BUILD/src" \
    -L"$CUDA/lib64" \
    -Wl,--no-as-needed \
    -Wl,--start-group -lncnn -lncnn_cuda_lib -Wl,--end-group \
    -lcudart -lcublas -lpthread -ldl \
    -Wl,-rpath,"$NCNN_BUILD/src" \
    -Wl,-rpath,"$CUDA/lib64"

echo "built: ${EVAL_BIN:-$HERE/ncnn_cuda_eval}"
ls -la "${EVAL_BIN:-$HERE/ncnn_cuda_eval}"
