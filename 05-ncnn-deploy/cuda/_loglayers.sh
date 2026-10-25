#!/usr/bin/env bash
# 打开 LOG_LAYERS 重新编译，用来看清每一层到底走没走 CUDA。
# 老 fork 的 README 自己给的诊断手法：跑一次网络，grep 出「没有走 cuda」的层。
# CPU/GPU 之间来回搬数据（ping-pong）是小图上 GPU 反而更慢的头号原因，
# 不看清这一步就只能靠猜。
set -eu
export PATH=/home/user/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64:${LD_LIBRARY_PATH:-}

NCNN=/home/user/ncnn-cuda
CUDA=/home/user/cuda

cd "$NCNN/build"
cmake -DNCNN_BUILD_TOOLS=OFF -DLOG_LAYERS=ON .. 2>&1 | tail -3
make -j"$(nproc)" 2>&1 | grep -E "error|Error|Built target ncnn$" | tail -8

# 评测程序也要带上同样的宏，否则日志开关不生效
g++ -O2 -std=c++11 -fopenmp -DLOG_LAYERS=1 \
    /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/ncnn_cuda_eval.cpp \
    /mnt/d/WorkBuddy/ShowBro/lab/04-libtorch-cuda/cpp/cifar10.cpp \
    -o /mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/ncnn_cuda_eval_log \
    -I/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda \
    -I/mnt/d/WorkBuddy/ShowBro/lab/04-libtorch-cuda/cpp \
    -I"$NCNN/src" -I"$NCNN/build/src" -I"$CUDA/include" \
    -L"$NCNN/build/src" -L"$CUDA/lib64" \
    -Wl,--start-group -lncnn -lncnn_cuda_lib -Wl,--end-group \
    -lcudart -lcublas -lpthread -ldl \
    -Wl,-rpath,"$NCNN/build/src" -Wl,-rpath,"$CUDA/lib64" 2>&1 | tail -5
echo "built ncnn_cuda_eval_log"
