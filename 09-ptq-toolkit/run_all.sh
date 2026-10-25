#!/usr/bin/env bash
# M2 一键跑通：三模型 × W8/W4 → bench → report → 四个验收脚本。
#
# 全程**只用 CPU**（device=cpu），不碰 GPU。理由两条：
#   1. GPU 让给用户
#   2. 重测会引入「机器负载」这个变量，让 golden test 不可复现
#      （08-report 已实测：同一份代码 idle 与占用状态差 21 倍）
#
# ⏱ 耗时参考（单核 CPU 上实测，约 10–15 分钟）：
#   ResNet-18  W8/W4   各约 3 分钟（全量 10000 张评测）
#   ResNet-50  W8/W4   各约 3 分钟（只取前 2000 张 —— 随机初始化模型全量无意义）
#   MobileNetV2 W8/W4  各约 1 分钟（模型小）
#
# 环境变量：
#   CALIB_N=128    校准样本数（M2 用 128 即可，量化语义对校准量不敏感）
#   EVAL_N=0       评测样本上限（0 = 由模型自己决定）
#   SKIP_BUILD=1   跳过 build，只跑验收
#
# ⚠️ 本脚本**不使用训练权重**（build 走模型默认初始化），只验证代码路径；
#    M2 真实精度在 results/m2_accept/（训练权重 + --weights 复现）。
#    跑完会把 root 的 build_*.json 刷回 ≈10% 的旧口径，不要拿它当 M2 报告。
set -eu
cd "$(dirname "$0")"
PY=$HOME/pyenv-torch/bin/python
CALIB_N=${CALIB_N:-128}
EVAL_N=${EVAL_N:-0}

echo "════════ GPU 未被本脚本占用（取证）════════"
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv 2>/dev/null || echo "(nvidia-smi 不可用)"
echo "↑ 若上面没有 python 进程，说明本轮没有我们的 GPU 计算进程"
echo ""

if [ "${SKIP_BUILD:-0}" != "1" ]; then
    echo "════ 1/6  build W8（三模型）════"
    $PY -m ptq_toolkit build --model all --w-bits 8 \
        --calib-n "$CALIB_N" --eval-n "$EVAL_N" --out-dir results --device cpu

    echo ""
    echo "════ 2/6  build W4（三模型）════"
    $PY -m ptq_toolkit build --model all --w-bits 4 \
        --calib-n "$CALIB_N" --eval-n "$EVAL_N" --out-dir results --device cpu
fi

echo ""
echo "════ 3/6  bench（引用 06/07 已有结果，不重测 GPU）════"
$PY -m ptq_toolkit bench --out-dir results

echo ""
echo "════ 4/6  report ════"
$PY -m ptq_toolkit report --out-dir results

echo ""
echo "════ 5/6  验收：量化算子（与 03 逐位对照）════"
$PY tests/test_quant_ops.py

echo ""
echo "════ 6/6  验收：模型 / 二进制格式 / golden ════"
$PY tests/test_models.py
$PY tests/check_binary_compat.py
$PY tests/test_golden.py

echo ""
echo "════════ 全部通过。M2 完成。════════"
