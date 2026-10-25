"""导出 TensorRT 要用的 ONNX（两种）。

为什么要导出两份：
  fp32 dynamic  -> PTQ 的输入，INT8 QDQ 图的底稿
  fp16          -> 直接给 TRT 建 fp16 engine

为什么不在这之后 set_flag(BuilderFlag.FP16)：TRT 11.4 的 python 绑定把这个枚举整组删了，
而且 set_flag(idx0) 和 cfg.flags |= 1 两条路实测都设不进去（读回永远是 64）。
唯一可行的办法就是让图本身是 Float16 typed。

用法:
    python export_onnx.py fp32
    python export_onnx.py fp16
    python export_onnx.py all
"""
import sys
from pathlib import Path

import torch

from paths import CKPT, PTQ_LAB, TRT_WORKDIR, ensure_workdir

sys.path.insert(0, str(PTQ_LAB))
import os  # noqa: E402

os.chdir(PTQ_LAB)
from src.model import load_fp32  # noqa: E402


def _export(model_path, half):
    m = load_fp32(ckpt=str(CKPT), num_classes=10).eval()
    x = torch.randn(4, 3, 32, 32)
    if half:
        m, x = m.half(), x.half()
    with torch.no_grad():
        torch.onnx.export(
            m, x, str(model_path),
            input_names=["input"], output_names=["logits"],
            opset_version=13,
            do_constant_folding=True,
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        )
    import onnx
    md = onnx.load(str(model_path))
    onnx.checker.check_model(md)
    et = md.graph.input[0].type.tensor_type.elem_type
    print(f"saved {model_path.name}: {model_path.stat().st_size/1e6:.1f} MB, "
          f"{len(md.graph.node)} nodes, input elem_type={et}"
          f"{' (Float16)' if et == 10 else ' (Float)'}")


def main():
    ensure_workdir()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("fp32", "all"):
        # 名字必须和 quantize_int8.py / trt_eval.py 的约定一致：fp32 -> *_fp32.onnx
        _export(TRT_WORKDIR / "resnet18_cifar10_fp32.onnx", half=False)
    if which in ("fp16", "all"):
        _export(TRT_WORKDIR / "resnet18_cifar10_fp16.onnx", half=True)


if __name__ == "__main__":
    main()
