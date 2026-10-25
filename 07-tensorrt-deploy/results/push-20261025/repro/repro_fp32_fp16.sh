#!/usr/bin/env bash
# push-20261025 · 补齐主表另外两行：fp32（1x1 折叠）与 fp16（1x1 折叠 + opt5 + DIRECT_IO）
#
# 为什么要补：
#   08-report 的新主表如果只有 int8 是新口径、fp32/fp16 还是旧 idle 口径，
#   那张表本身就是混口径的。所以三行都必须跑同一套 protocol：
#   batch=512 / opt3（fp32）或 opt5+DIRECT_IO（fp16）/ preboost=100 / steady。
#
# 图全部在 repro/ 下重新生成，engine 目录空 → 必然冷构建。
set -euo pipefail

REPO=/mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy
P=/home/user/pyenv-torch/bin/python
cd "$REPO"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
R=results/push-20261025/repro
mkdir -p "$R/work" "$R/engines"
exec > >(tee "$R/repro_fp32_fp16.txt") 2>&1

step() { echo; echo "### $*"; }

step "fp32：折叠后的图已经在上一步生成（$R/resnet18_cifar10_fp32_c1x1.onnx），直接冷构建 + 测"
bash scripts/push_one.sh "$R/resnet18_cifar10_fp32_c1x1.onnx" 512 fp32_c1x1_pb \
    --preboost 100 \
    --jsonl "$R/bench.jsonl" --engine-dir "$R/engines" \
    --timing-cache "$R/timing.cache"

step "fp16：从 ONNX 重新导出 fp16 底稿并折叠（与 07 的图对照 sha256）"
TRT_WORKDIR="$PWD/$R/work" "$P" scripts/export_onnx.py fp16
"$P" scripts/push_optimize_onnx.py \
    --in  "$R/work/resnet18_cifar10_fp16.onnx" \
    --out "$R/resnet18_cifar10_fp16_c1x1.onnx"
sha256sum "$R/resnet18_cifar10_fp16_c1x1.onnx" \
          results/push-20261025/resnet18_cifar10_fp16_c1x1.onnx \
    | tee "$R/sha256_fp16.txt"

step "fp16：冷构建 opt5 + DIRECT_IO 的 engine 并测（复核 0.01068 / 87.18%）"
bash scripts/push_one.sh "$R/resnet18_cifar10_fp16_c1x1.onnx" 512 repro_fp16_c1x1_pb \
    --opt 5 --direct-io --preboost 100 \
    --jsonl "$R/bench.jsonl" --engine-dir "$R/engines" \
    --timing-cache "$R/timing.cache"

step "SHA256（本轮新增的 engine / 图）"
{
  sha256sum "$R"/engines/*.trt
  sha256sum "$R/resnet18_cifar10_fp16_c1x1.onnx"
} | tee "$R/sha256_fp32_fp16.txt"
