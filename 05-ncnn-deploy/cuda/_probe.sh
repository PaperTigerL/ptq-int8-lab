#!/usr/bin/env bash
# 编译并运行 GPU 探针（严格遵守：命令一律写脚本，别写 wsl -lc 内联）。
set -eu
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64:${LD_LIBRARY_PATH:-}

HERE=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda
NCNN=/home/user/ncnn-cuda
CUDA=/home/user/cuda

g++ -O2 -std=c++11 -fopenmp "$HERE/probe_gpu.cpp" -o "$HERE/probe_gpu" \
    -I"$NCNN/src" -I"$NCNN/build/src" -I"$CUDA/include" \
    -L"$NCNN/build/src" -L"$CUDA/lib64" \
    -Wl,--start-group -lncnn -lncnn_cuda_lib -Wl,--end-group \
    -lcudart -lcublas -lpthread -ldl \
    -Wl,-rpath,"$NCNN/build/src" -Wl,-rpath,"$CUDA/lib64"

"$HERE/probe_gpu"
