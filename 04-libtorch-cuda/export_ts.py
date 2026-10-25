import os
import sys
from pathlib import Path

import torch
import torchvision.models as tv

ROOT = Path("/mnt/d/WorkBuddy/ShowBro")
PTQ = str(ROOT / "lab" / "03-ptq-lab")
sys.path.insert(0, PTQ)
os.chdir(PTQ)

from src.model import load_fp32  # noqa: E402

CKPT = ROOT / "paper-code" / "cifar_resnet18_50ep.pth"
OUT = Path("/mnt/d/WorkBuddy/_scratch/libtorch_cuda/resnet18_cifar10.pt")

m = load_fp32(ckpt=str(CKPT), num_classes=10)
sd = torch.load(CKPT, map_location="cpu")
m.load_state_dict(sd)
m.eval()

# 直接 trace（模型里没有动态控制流）
example = torch.randn(1, 3, 32, 32)
traced = torch.jit.trace(m, example)
traced = traced.eval()

OUT.parent.mkdir(parents=True, exist_ok=True)
traced.save(str(OUT))
print("saved:", OUT, OUT.stat().st_size, "bytes")

# 自检：C++ 端加载前后输出一致
traced_cpu = torch.jit.load(str(OUT)).eval().cpu()
x = torch.randn(4, 3, 32, 32)
with torch.no_grad():
    a = m(x)
    b = traced_cpu(x)
print("max diff:", (a - b).abs().max().item())
