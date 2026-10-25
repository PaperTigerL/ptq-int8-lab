#!/usr/bin/env bash
# 编译 ncnn-with-cuda（社区 CUDA 分支）。
# 注意：WSL 命令经 Git Bash 转发会吃掉 $VAR，所以必须写成脚本文件。
set -u
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64:${LD_LIBRARY_PATH:-}

cd /home/user/ncnn-cuda
mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_VULKAN=OFF \
      -DNCNN_CUDA=ON \
      -DNCNN_BUILD_EXAMPLES=OFF \
      -DNCNN_BUILD_TOOLS=ON \
      -DNCNN_BUILD_BENCHMARK=OFF \
      -DCMAKE_CUDA_COMPILER=/home/user/cuda/bin/nvcc \
      .. 2>&1 | tail -25

echo "=== make ==="
make -j"$(nproc)" 2>&1 | tail -40
echo "=== 产物 ==="
ls -la src/libncnn.a bench/libncnn.so 2>/dev/null
ls tools/*.cpp 2>/dev/null | head -3
ls tools/ncnn2table tools/ncnn2int8 2>/dev/null
