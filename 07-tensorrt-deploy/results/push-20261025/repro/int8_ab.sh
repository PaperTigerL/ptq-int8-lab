#!/usr/bin/env bash
# 决定性 A/B：同场（同 2 分钟窗口内）对比
#   A = 07 原轮 int8 engine（f78b02ec）
#   B = 复现重建的 int8 engine（78b88e16，存成 int8_repro_build2.trt）
# 如果 B 现在也快 → 之前 0.0089–0.0092 是瞬时干扰；如果 B 稳定慢 → 是 engine 的差异。
set -uo pipefail
REPO=/mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy
cd "$REPO"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
P=/home/user/pyenv-torch/bin/python
R=results/push-20261025/repro
J=$R/bench_ab.jsonl
ONNX=$R/resnet18_cifar10_int8_c1x1.onnx
exec > >(tee "$R/int8_ab.txt") 2>&1

run() {  # run <tag> <engine>
  flock -x /tmp/ptq-gpu.lock -c "cd $REPO && LD_LIBRARY_PATH=/usr/lib/wsl/lib $P scripts/push_bench.py \
    --onnx $ONNX --batch 512 --tag $1 --skip 1000 --preboost 100 \
    --engine-path $2 --jsonl $J" | grep -E "dataset round|steady:|^=="
}

echo "### A1 原轮 engine（f78b02ec）"
run ab_orig_a1 work/push-20261025/engines/resnet18_cifar10_int8_c1x1__b512_o3_default_w1.0_aNone_xNone_d0_f78b02ece7.trt
echo "### B1 复现 build2（78b88e16）"
run ab_repro_b1 "$R/engines/int8_repro_build2.trt"
echo "### A2 原轮 engine 再测一次"
run ab_orig_a2 work/push-20261025/engines/resnet18_cifar10_int8_c1x1__b512_o3_default_w1.0_aNone_xNone_d0_f78b02ece7.trt
echo "### B2 复现 build2 再测一次"
run ab_repro_b2 "$R/engines/int8_repro_build2.trt"
