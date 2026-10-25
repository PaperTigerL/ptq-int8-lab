#!/usr/bin/env bash
# 冒烟测试：从零走一遍导出 → 量化，确认交付的脚本不是只能在我用的那个目录里跑。
# 注意：WSL 命令经 Git Bash 转发会吃掉 $VAR，所以必须写成脚本，不能写内联命令。
set -eu
cd /mnt/d/WorkBuddy/ShowBro/lab/07-tensorrt-deploy/scripts
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
PY=$HOME/pyenv-torch/bin/python
export TRT_WORKDIR=/mnt/d/WorkBuddy/_scratch/smoke_work

rm -rf "$TRT_WORKDIR"
mkdir -p "$TRT_WORKDIR"

echo "== export fp32/fp16 =="
$PY export_onnx.py all

echo "== quantize int8 =="
$PY quantize_int8.py --calib-n 1000

echo "== 产物 =="
ls -la "$TRT_WORKDIR"
