#!/usr/bin/env bash
# 增量重建：只重编比 .o 新的 .cu，然后重链所有主机侧程序。
# （build.sh 每次都全量编 int8_kernels.cu，一次约 4 分钟；迭代时用这个。）
set -e
cd "$(dirname "$0")"
CUDA_HOME="$HOME/cuda"
LIBTORCH="$HOME/libtorch"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LIBTORCH/lib:$CUDA_HOME/lib64
INC="-I. -I$LIBTORCH/include -I$LIBTORCH/include/torch/csrc/api/include -I$CUDA_HOME/include"
LNK="-L$LIBTORCH/lib -Wl,--no-as-needed -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda -L$CUDA_HOME/lib64 -lcublas -lcudart -Wl,-rpath,$LIBTORCH/lib -Wl,-rpath,$CUDA_HOME/lib64 -Wl,-rpath,/usr/lib/wsl/lib"

for pair in "int8_kernels.cu build/int8_kernels.o" "implicit_conv.cu build/implicit_conv.o"; do
    set -- $pair
    if [ ! -f "$2" ] || [ "$1" -nt "$2" ]; then
        echo "[rebuild] $1"
        nice -n 19 $CUDA_HOME/bin/nvcc -O3 -std=c++17 -arch=sm_75 -c "$1" -o "$2"
    else
        echo "[skip] $1 (up to date)"
    fi
done

echo "[host] main.cpp / cifar10.cpp / 链接"
nice -n 19 g++ -O3 -std=c++17 -c main.cpp -o build/main.o $INC
nice -n 19 g++ -O3 -std=c++17 -c cifar10.cpp -o build/cifar10.o -I.
g++ build/main.o build/int8_kernels.o build/implicit_conv.o build/cifar10.o -o build/gpu_int8 $LNK

echo "[tool] bench_layers / bench_streams / check_layer"
bash build_bench_layers.sh
bash build_streams.sh
nice -n 19 g++ -O2 -std=c++17 check_layer.cpp build/int8_kernels.o build/implicit_conv.o -o build/check_layer $INC $LNK
echo "RELINKED"
