#!/usr/bin/env bash
set -u
cd /tmp
git -C /home/user/ncnn-cuda show 8918acf:src/layer/cuda/implicit_conv_cuda.cu > old_impl.cu
cp /home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.cu smem_impl.cu
for f in old_impl smem_impl; do
  echo "=== $f ==="
  /home/user/cuda/bin/nvcc -c "$f.cu" -o "/tmp/$f.o" -DNCNN_CUDA=1 \
    -Xcompiler -O3 -DNDEBUG -arch=sm_75 \
    -I/home/user/cuda/include -I/home/user/ncnn-cuda/src \
    -I/home/user/ncnn-cuda/build-ptq/src -I/home/user/ncnn-cuda/src/layer \
    -I/home/user/ncnn-cuda/src/layer/cuda \
    -Xptxas -v 2>&1 | grep -E "implicit_conv_int8_kernel|registers|spill"
done
