#!/usr/bin/env bash
# 全套验证：先跑四个探针（数值正确性），再跑全量评测。
#
# ⭐ 为什么要一个统一入口：
#   这一轮踩的坑里，至少三次是「探针自己的 bug 被误读成 kernel 的 bug」
#   （int8 编码溢出、move 后用已移空的 vector、手写声明参数个数不对）。
#   把验证收在一个脚本里，意味着每次改动后**一次跑完**，
#   而不是靠人记得该跑哪个。
set -e
cd "$(dirname "$0")"

CUDA_HOME=${CUDA_HOME:-$HOME/cuda}
LIBTORCH=${LIBTORCH:-$HOME/libtorch}
CUBLAS_DIR=$CUDA_HOME/lib64
DATA=${DATA:-/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw}
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LIBTORCH/lib:$CUBLAS_DIR

mkdir -p build
bash build.sh > /dev/null
bash build_check.sh > /dev/null

FAIL=0
run() {
    echo ""
    echo "──── $* ────"
    if "$@"; then :; else echo "  [FAIL] $*"; FAIL=1; fi
}

echo "════ 探针：数值正确性（从最小到完整）════"
g++ -O2 -std=c++17 probe_gemm.cpp -o build/probe_gemm -I. \
    -I$LIBTORCH/include -I$LIBTORCH/include/torch/csrc/api/include -I$CUDA_HOME/include \
    -L$LIBTORCH/lib -Wl,--no-as-needed -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda \
    -L$CUBLAS_DIR -lcudart -Wl,-rpath,$LIBTORCH/lib -Wl,-rpath,$CUBLAS_DIR
for p in probe_gemm probe_im2col probe_im2col_quant probe_zp; do
    g++ -O2 -std=c++17 $p.cpp build/int8_kernels.o build/implicit_conv.o -o build/$p -I. \
        -I$LIBTORCH/include -I$LIBTORCH/include/torch/csrc/api/include -I$CUDA_HOME/include \
        -L$LIBTORCH/lib -Wl,--no-as-needed -ltorch -ltorch_cuda -ltorch_cpu -lc10 -lc10_cuda \
        -L$CUBLAS_DIR -lcublas -lcudart -Wl,-rpath,$LIBTORCH/lib -Wl,-rpath,$CUBLAS_DIR
    run ./build/$p
done

echo ""
echo "════ 单层数值校验（vs CPU 参考实现）════"
run ./build/check_layer resnet18_folded.bin calib_minmax.bin 2 --n 4 --quiet
run ./build/check_layer resnet18_folded.bin calib_minmax.bin 9 --n 4 --quiet

echo ""
echo "════ 全量评测（10000 张）════"
# ⚠️ batch=500 是这台卡的实测最优点（250→0.119, 500→0.094, 1000→更差）
# batch 越大 im2col 矩阵越大（batch×1024×576 int8），4 GB 显存放不下 batch=2000。
for wb in "8 8" "6 8" "4 8"; do
    set -- $wb
    echo "--- W${1}A${2} ---"
    ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" \
        --limit 10000 --batch 500 --w-bits "$1" --a-bits "$2" | tail -4
done

echo ""
if [ $FAIL -eq 0 ]; then
    echo "════ 全部探针通过 ════"
else
    echo "════ 有探针失败 ════"
    exit 1
fi
