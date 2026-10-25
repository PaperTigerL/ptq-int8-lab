#!/usr/bin/env bash
# 测量噪声：同一个 int8 engine（不重建）连续测 3 次（每次约 9 s），
# 用来区分「engine/图 的差异」与「当轮机器状态的差异」。
#
# 背景：int8 第二条构建测到 0.00733，紧接着同一个 engine 复测却掉到 0.00920（慢 25%），
#   说明这台机器上偶发的 GPU 争用/功耗状态变化能造成 >20% 的偏差 ——
#   复现验收必须看多次测量的中位数，不能用单次。
set -uo pipefail
REPO=/mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy
cd "$REPO"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
R=results/push-20261025/repro
J=$R/bench_int8v.jsonl
for i in 1 2 3; do
  echo "### 第 $i 次（不重建）"
  bash scripts/push_one.sh "$R/resnet18_cifar10_int8_c1x1.onnx" 512 "int8_probe_$i"       --skip 1000 --preboost 100 --jsonl "$J"       --engine-dir "$R/engines" --timing-cache "$R/timing.cache" 2>&1     | grep -E "dataset round|steady:|^==|GATE_UTIL|[gpu]"
done
