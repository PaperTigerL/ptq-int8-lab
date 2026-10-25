#!/usr/bin/env bash
# 最终测量：候选配置跑 20 轮取最小。机器被占时单轮不可信，多轮最小才收敛。
set -u
cd /mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy/scripts
export TRT_WORKDIR=/mnt/d/WorkBuddy/_scratch/libtorch_cuda
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
P=$HOME/pyenv-torch/bin/python
OUT=/mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy/results/final.txt
mkdir -p "$(dirname "$OUT")"
echo "# final $(date '+%F %T') repeat=20，报告多轮最小值" > "$OUT"

run() {
    echo "--- $1 batch $2 skip $3" | tee -a "$OUT"
    $P trt_eval.py --model "$1" --batch "$2" --skip "$3" --repeat 20 2>&1 \
        | tee -a "$OUT" | grep -E "最优|精度 "
}

run fp16 512  0
run fp16 2048 0
run fp16 4096 0
run int8 512  1000
run fp32 512  0
echo "写入 $OUT"
