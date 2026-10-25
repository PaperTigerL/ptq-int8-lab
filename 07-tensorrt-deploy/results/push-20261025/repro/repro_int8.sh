#!/usr/bin/env bash
# push-20261025 · 独立复现 int8 c1x1：从 ONNX 干净重建 engine，再跑精度 + 延迟
#
# 为什么要有这一轮：07 原轮的 0.00744 ms/img / 87.22% 必须能被第三方从图重放到
# 同一量级，否则它只是“某次跑出来的数”，不是交付物。
#
# 口径（与 07 原轮逐项一致，**不做任何调优**）：
#   batch=512, opt=3, tactic=default, workspace=1 GB, static shape
#   skip=1000（前 1000 张是校准集，不参与评测）
#   精度 = 同一次 dataset pass 上算（9000 张）
#   延迟 = dataset（逐批 H2D→execute→sync，含 preboost 100 顶 clock）
#        + steady（同址 buffer 背靠背 enqueue）
#
# 与 07 原轮的**唯一**差别：所有 ONNX 在 repro/ 下重新生成、engine 目录是空的
#   → 必然走冷构建，不可能复用 07 的 engine 缓存（这才是“干净重建”）。
#
# 用法：bash results/push-20261025/repro/repro_int8.sh
set -euo pipefail

REPO=/mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy
LAB=/mnt/d/WorkBuddy/ShowBro/lab
P=/home/user/pyenv-torch/bin/python
cd "$REPO"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib

R=results/push-20261025/repro
mkdir -p "$R/work" "$R/engines"
# 整轮日志：既进终端也进文件（.log 被根 .gitignore 忽略，所以用 .txt）
exec > >(tee "$R/repro_int8.txt") 2>&1

step() { echo; echo "### $*"; }

step "0/5 环境快照"
{
  date -Is
  echo "git HEAD: $(git -C "$LAB" rev-parse --short=7 HEAD)"
  echo "python:   $("$P" -V 2>&1)"
  "$P" - <<'PY'
import torch, tensorrt, onnx, onnxruntime
print("torch", torch.__version__, "| tensorrt", tensorrt.__version__)
print("onnx", onnx.__version__, "| onnxruntime", onnxruntime.__version__)
print("gpu:", torch.cuda.get_device_name(0))
PY
  nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,driver_version --format=csv,noheader
} | tee "$R/env.txt"

step "1/5 导出 fp32 ONNX 到底稿目录（TRT_WORKDIR 指到 repro/work）"
TRT_WORKDIR="$PWD/$R/work" "$P" scripts/export_onnx.py fp32

step "2/5 语义等价优化：1x1 空间上的 3x3 卷积折成 1x1（带 ORT 数值校验）"
"$P" scripts/push_optimize_onnx.py \
    --in  "$R/work/resnet18_cifar10_fp32.onnx" \
    --out "$R/resnet18_cifar10_fp32_c1x1.onnx" --verify-ort

step "3/5 int8 QDQ 量化（复用 quantize_int8 的 CifarReader / strip_int32_bias_dq）"
"$P" scripts/push_quantize_variant.py \
    --src "$R/resnet18_cifar10_fp32_c1x1.onnx" \
    --dst "$R/resnet18_cifar10_int8_c1x1.onnx"

# 4 GB 卡上 int8 冷构建 ~11 分钟；先跑小样本（2000 张）把 engine 建出来，
# 第二轮复用同一 engine 跑全量 9000 张 + 逐层 profile。
step "4/5a 冷构建 int8 engine + 精度/延迟（2000 张）"
bash scripts/push_one.sh "$R/resnet18_cifar10_int8_c1x1.onnx" 512 repro_int8_l2000 \
    --skip 1000 --limit 2000 --preboost 100 \
    --jsonl "$R/bench.jsonl" --engine-dir "$R/engines" \
    --timing-cache "$R/timing.cache"

step "4/5b 复用同一 engine：全量 9000 张 + 逐层 profile"
bash scripts/push_one.sh "$R/resnet18_cifar10_int8_c1x1.onnx" 512 repro_int8_full9000 \
    --skip 1000 --preboost 100 --profile \
    --jsonl "$R/bench.jsonl" --engine-dir "$R/engines" \
    --timing-cache "$R/timing.cache"

step "5/5 SHA256（复现产物 + 07 原轮的同名产物，用于“图是否同一份”的对账）"
{
  echo "# 本轮重新生成的图与 engine"
  sha256sum "$R/work/resnet18_cifar10_fp32.onnx" \
            "$R/resnet18_cifar10_fp32_c1x1.onnx" \
            "$R/resnet18_cifar10_int8_c1x1.onnx" \
            "$R"/engines/*.trt
  echo
  echo "# 07 原轮留在 results/push-20261025/ 下的同名图（不重写，只做对照）"
  sha256sum results/push-20261025/resnet18_cifar10_fp32_c1x1.onnx \
            results/push-20261025/resnet18_cifar10_int8_c1x1.onnx
  echo
  echo "# 生成这些图/engine 的脚本"
  sha256sum scripts/push_bench.py scripts/push_optimize_onnx.py \
            scripts/push_quantize_variant.py scripts/push_one.sh \
            results/push-20261025/repro/repro_int8.sh
  echo
  echo "git HEAD $(git -C "$LAB" rev-parse HEAD)"
} | tee "$R/sha256.txt"

step "对账：复现值 vs 07 原轮（±3% 延迟 / ±0.5 pp 精度）"
"$P" results/push-20261025/repro/compare_repro.py --tee "$R/compare.txt"
