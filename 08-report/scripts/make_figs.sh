#!/usr/bin/env bash
# 生成 08-report 的全部图表与汇总表。
#
# 从**空 work/ 重跑也能工作**：
#   make_figs.sh 只依赖 results/*.txt 和 results/gemm_tops.csv，
#   不依赖任何 TRT engine 或中间产物 —— 图是报告的派生物，
#   数据源永远是 results，不是缓存。
#
# 唯一的外部依赖：matplotlib（~/pyenv-torch 环境里已装）。
# 若缺失会提示装哪个包，不静默失败。
set -eu
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
PY=$HOME/pyenv-torch/bin/python

if ! $PY -c "import matplotlib" 2>/dev/null; then
    echo "缺 matplotlib，装一下：$PY -m pip install matplotlib"
    exit 1
fi

$PY scripts/make_figs.py

echo ""
echo "产出："
ls -la figs/
echo ""
echo "表格："
cat results/summary.csv
