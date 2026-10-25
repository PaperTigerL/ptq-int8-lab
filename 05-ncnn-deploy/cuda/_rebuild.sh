#!/usr/bin/env bash
# 干净重建 ncnn-with-cuda（关掉 LOG_LAYERS）。
#
# 为什么必须重建：上次为了诊断「层走没走 CUDA」，跑过一次
#   cmake -DLOG_LAYERS=ON ..
# CMakeLists 第 17-20 行把 LOG_LAYERS 直接拼进全局 CMAKE_CXX_FLAGS
# （-DLOG_LAYERS=1），于是整个库每一层的 forward 都会 fprintf 一行日志。
# 带着这个库测出来的延迟没有任何意义，必须关掉重编。
#
# 注意：WSL 命令经 Git Bash 转发会吃掉 $VAR 并且引号嵌套容易炸，所以一律写成脚本文件。
set -u
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64:${LD_LIBRARY_PATH:-}

python3 /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/patch_ncnn_cuda.py

rm -rf /home/user/ncnn-cuda/build
mkdir -p /home/user/ncnn-cuda/build
cd /home/user/ncnn-cuda/build

cmake -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_CUDA=ON \
      -DNCNN_VULKAN=OFF \
      -DNCNN_BUILD_TOOLS=OFF \
      -DNCNN_OPENMP=ON \
      -DLOG_LAYERS=OFF \
      -DCUDA_TOOLKIT_ROOT_DIR=/home/user/cuda \
      .. 2>&1 | tail -15

make -j"$(nproc)" 2>&1 | grep -E "error|Error|Built target" | tail -20
echo "=== 产物 ==="
ls -la /home/user/ncnn-cuda/build/src/libncnn.a /home/user/ncnn-cuda/build/src/libncnn_cuda_lib.a 2>&1
