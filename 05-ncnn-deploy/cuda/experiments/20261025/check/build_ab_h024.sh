#!/usr/bin/env bash
# Build old (8918acf) and smem variants of the eval binary for A/B timing.
set -euo pipefail
SRC=/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.cu
LAB=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/user/cuda/lib64

cp "$SRC" /tmp/implicit_conv_smem.cu
cd /home/user/ncnn-cuda
git checkout -- "$SRC"
cmake --build build-ptq -j8 >/tmp/build_old.log 2>&1
cd "$LAB"
NCNN_BUILD_DIR=/home/user/ncnn-cuda/build-ptq \
EVAL_SRC="$LAB/ncnn_cuda_eval_2000c.cpp" EVAL_BIN="$LAB/ncnn_cuda_eval_2000c_old" \
  bash build.sh >/tmp/bold.log 2>&1

cp /tmp/implicit_conv_smem.cu "$SRC"
cd /home/user/ncnn-cuda
cmake --build build-ptq -j8 >/tmp/build_smem.log 2>&1
cd "$LAB"
NCNN_BUILD_DIR=/home/user/ncnn-cuda/build-ptq \
EVAL_SRC="$LAB/ncnn_cuda_eval_2000c.cpp" EVAL_BIN="$LAB/ncnn_cuda_eval_2000c" \
  bash build.sh >/tmp/bsmem.log 2>&1
NCNN_BUILD_DIR=/home/user/ncnn-cuda/build-ptq \
EVAL_SRC="$LAB/ncnn_cuda_profile_layers.cpp" EVAL_BIN="$LAB/ncnn_cuda_profile_layers" \
  bash build.sh >/tmp/bprof.log 2>&1

echo "old: $(ls -la "$LAB/ncnn_cuda_eval_2000c_old" | awk '{print $5}')"
echo "smem: $(ls -la "$LAB/ncnn_cuda_eval_2000c" | awk '{print $5}')"
