#!/usr/bin/env bash
# PTQ 实验平台 · 环境自检（WSL 专用）
#
# 这个脚本只做检查，不替你装东西 —— 缺什么它会告诉你该敲哪条命令。
# 已经跑通过的环境再跑一次也能用来确认没坏。
set -u

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$LAB_DIR" || exit 1
PY="$HOME/pyenv-torch/bin/python"

ok=0
bad=0
note() { printf "  %-10s %s\n" "$1" "$2"; }
pass() { echo "[OK]   $1"; ok=$((ok+1)); }
fail() { echo "[MISS] $1"; echo "       修法: $2"; bad=$((bad+1)); }

echo "============================================================"
echo "PTQ 实验平台 · 环境自检"
echo "============================================================"

# 1. 是否在 WSL
if grep -qi microsoft /proc/version 2>/dev/null; then
    pass "运行在 WSL 里"
else
    fail "没检测到 WSL" "本平台依赖 WSL 的 torch 环境，请在 wsl 里执行"
fi

# 2. python 环境
if [ -x "$PY" ]; then
    pass "torch 环境: $PY ($($PY -V 2>&1))"
else
    fail "找不到 ~/pyenv-torch/bin/python" \
         "python3 -m venv ~/pyenv-torch && ~/pyenv-torch/bin/pip install torch torchvision"
fi

# 3. 依赖与 GPU
if [ -x "$PY" ]; then
    "$PY" - <<'PYEOF'
import importlib, sys
mods = ["torch", "torchvision", "numpy"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print("[OK]   %-14s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        print("[MISS] %-14s %s" % (m, e))
try:
    import torch
    print("[OK]   CUDA available = %s" % torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("[OK]   GPU %s  %.2f GB" % (p.name, p.total_memory/1024**3))
    else:
        print("[WARN] 没有 GPU，实验会退到 CPU，每个实验慢 5-10 倍但仍可跑")
except Exception as e:
    print("[MISS] torch import: %s" % e)
PYEOF
fi

# 4. 数据集
DATA="/mnt/d/WorkBuddy/ShowBro/paper-code/data/cifar-10-batches-py"
if [ -d "$DATA" ]; then
    n=$(ls "$DATA" 2>/dev/null | wc -l)
    pass "CIFAR-10 数据 ($n 个文件)"
else
    fail "CIFAR-10 数据不在 $DATA" \
         "python -c \"import torchvision; torchvision.datasets.CIFAR10('$DATA', download=True)\""
fi

# 5. 权重
CK="/mnt/d/WorkBuddy/ShowBro/paper-code/cifar_resnet18_50ep.pth"
if [ -f "$CK" ]; then
    sz=$(du -m "$CK" | cut -f1)
    pass "主基线权重 (${sz}MB, 实测 87.16%)"
else
    fail "主基线权重不在 $CK" \
         "先跑 paper-code/train_baseline.py 训一个（GTX 1650 Ti 上约 40 分钟）"
fi
CKW="/mnt/d/WorkBuddy/ShowBro/paper-code/cifar_resnet18.pth"
if [ -f "$CKW" ]; then
    pass "弱基线权重（仅 e1 做对照用）"
else
    echo "[INFO] 弱基线权重缺失，e1 的对照会跳过（不影响主流程）"
fi

# 6. 目录
mkdir -p results
pass "results/ 目录就绪"

echo "============================================================"
echo "通过 $ok 项，缺失 $bad 项"
if [ $bad -eq 0 ]; then
    echo "环境就绪。下一步:  ./run.sh e1"
else
    echo "先把上面 [MISS] 的项补齐再跑实验。"
fi
echo "============================================================"
