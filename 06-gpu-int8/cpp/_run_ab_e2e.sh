#!/usr/bin/env bash
# 端到端同场 A/B：全线 im2col+cublas / 默认按层混合 / 全线 implicit。
# 口径：batch=512、100 次预热 + 60 次取最快（main.cpp 内的稳态循环），
#       每个配置前后各打一次 GPU 时钟。外部用 flock 包裹。
set -u
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/libtorch/lib:$HOME/cuda/lib64
DATA=/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw
LOG=../results/ab_e2e_20261025.log
: > "$LOG"

clock() { nvidia-smi --query-gpu=clocks.current.graphics,memory.used --format=csv,noheader; }

run() {
  local tag="$1"; shift
  echo "" | tee -a "$LOG"
  echo "===== $tag =====" | tee -a "$LOG"
  echo "clock(start): $(clock)" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG" | grep -E "\[impl\]|\[implicit\]|GPU 显存|精度|稳态|加速" | sed "s/^/  /"
  echo "clock(end):   $(clock)" | tee -a "$LOG"
}

# ---- 1) baseline：全线 im2col+cublas（ROOFLINE §10 的 0.0261 那条路）----
run "A0 baseline cublas batch512（limit 512）" \
    env -u LAYER_IMPL -u LAYER_IMPL_JSON USE_IMPLICIT_GEMM=0 \
    ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" --limit 512 --batch 512
run "A0b baseline cublas 10000 张（batch500，精度）" \
    env -u LAYER_IMPL -u LAYER_IMPL_JSON USE_IMPLICIT_GEMM=0 \
    ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" --limit 10000 --batch 500

# ---- 2) 新实现：默认按层混合（层选择表）----
run "B1 mixed(表) batch512（limit 512）" \
    env -u USE_IMPLICIT_GEMM -u LAYER_IMPL -u LAYER_IMPL_JSON \
    ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" --limit 512 --batch 512
run "B1b mixed(表) 10000 张（batch500，精度）" \
    env -u USE_IMPLICIT_GEMM -u LAYER_IMPL -u LAYER_IMPL_JSON \
    ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" --limit 10000 --batch 500

# ---- 3) A/B 对照：全线 implicit（改动前的默认）----
run "A1 all-implicit batch512（limit 512）" \
    env -u LAYER_IMPL -u LAYER_IMPL_JSON USE_IMPLICIT_GEMM=1 \
    ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" --limit 512 --batch 512

# ---- 4) 复测 mixed：确认不是时钟/顺序偶然 ----
run "B2 mixed(表) 复测 batch512（limit 512）" \
    env -u USE_IMPLICIT_GEMM -u LAYER_IMPL -u LAYER_IMPL_JSON \
    ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" --limit 512 --batch 512

# ---- 5) mixed 逐层计时（PROF_LAYERS，只用于看分布）----
run "B3 mixed(表) PROF_LAYERS batch512（limit 512）" \
    env -u LAYER_IMPL -u LAYER_IMPL_JSON PROF_LAYERS=1 \
    ./build/gpu_int8 resnet18_folded.bin calib_minmax.bin "$DATA" --limit 512 --batch 512

echo "" | tee -a "$LOG"
echo "==== clock final ====" | tee -a "$LOG"
clock | tee -a "$LOG"
