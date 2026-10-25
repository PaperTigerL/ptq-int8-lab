# -*- coding: utf-8 -*-
"""把 PyTorch 侧的 FP32 基线模型导出成 ONNX，供 onnx2ncnn 转换。

为什么要单独走这一步
--------------------
03-ptq-lab 所有精度数字都是在 PyTorch 里测的。要回答「这些结论在 C++ 引擎里
还成立吗」，第一步必须把**同一个权重文件**搬进 ncnn。ONNX 是唯一中间载体。

口径约定（不可随意改，否则和 ptq-lab 的数字对不上）
------------------------------------------------
1. 用的是 paper-code/ckpts/ 下那个**未经任何量化处理**的 FP32 权重。
   不是 BN 折叠后的、不是量化后的 —— ncnn 有自己的 Conv+BN 融合（ncnnoptimize），
   让它自己融，才能测出「ncnn 的融合 + PyTorch 的融合」是否有差异。
2. 输入是**已经归一化过**的 (1,3,32,32) 张量（ToTensor + CIFAR mean/std）。
   归一化在 C++ runner 里做，模型本身不掺和 —— 这样模型才是干净的、
   可和其他框架对比的。
3. opset 固定 11。ResNet-18 用到的算子（Conv/Relu/MaxPool/Add/GlobalAveragePool/
   Flatten/Gemm）在 opset 11 里全部稳定，ncnn 的 onnx2ncnn 对它的支持也最成熟。
   换 opset 属于「无收益的口径变更」，不做。

用法（WSL）
----------
    ~/pyenv-torch/bin/python export_onnx.py
    ~/pyenv-torch/bin/python export_onnx.py --dataset cifar100 --arch resnet18
"""
import argparse
import os
import sys

import torch

# ---------------------------------------------------------------- 路径
# 05-ncnn-deploy 和 03-ptq-lab 是平级目录，paper-code 在更上一层。
# 这三个常量写死是有意的：脚本只在 WSL 里跑，路径不会漂。
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
SHOWBRO = os.path.dirname(os.path.dirname(LAB))          # .../ShowBro
PAPER_CODE = os.path.join(SHOWBRO, "paper-code")
MODEL_DIR = os.path.join(LAB, "models")

DATASET_SPECS = {
    "cifar10": {"num_classes": 10,
                "mean": (0.4914, 0.4822, 0.4465),
                "std": (0.2470, 0.2435, 0.2616)},
    "cifar100": {"num_classes": 100,
                 "mean": (0.5071, 0.4865, 0.4409),
                 "std": (0.2673, 0.2564, 0.2762)},
}

# 和 03-ptq-lab/config.py 的 _CKPT_LEGACY 保持一致：
# cifar10+resnet18 走历史路径（87.16% 的来源，不能挪），其余在 ckpts/ 下。
LEGACY_CKPT = {("cifar10", "resnet18"): "cifar_resnet18_50ep.pth"}


def ckpt_path(dataset, arch):
    if (dataset, arch) in LEGACY_CKPT:
        return os.path.join(PAPER_CODE, LEGACY_CKPT[(dataset, arch)])
    return os.path.join(PAPER_CODE, "ckpts", "%s_%s_50ep.pth" % (dataset, arch))


def build_model(dataset, arch):
    from torchvision.models import resnet18, mobilenet_v2
    n = DATASET_SPECS[dataset]["num_classes"]
    if arch == "resnet18":
        return resnet18(num_classes=n)
    if arch == "mobilenetv2":
        return mobilenet_v2(num_classes=n)
    raise ValueError("未知 arch %r" % arch)


def quantize_weights(model, w_bits):
    """fold BN 后把每个 Conv2d 的权重量化到 w_bits 网格（仍以 fp32 存储）。

    为什么必须先 fold 再量化
    -----------------------
    BN 的 gamma/beta 会把权重的数值范围整体缩放，量化网格对不上。
    03-ptq-lab 的所有数字都是「fold BN 之后量化」得到的，这里必须同口径，
    否则两边的 W4 不是同一个 W4。

    为什么量化后仍存 fp32
    --------------------
    ncnn2int8 会用表里的 127/absmax 把 fp32 权重压进 int8 容器。只要权重值
    本身只落在 2^(w_bits-1) 个离散点上，压完就是等价的 sub-8bit 表示 ——
    数值语义完全一致，只是暂时没有体积/带宽收益（打包存储留到 P2）。
    """
    sys.path.insert(0, os.path.join(SHOWBRO, "lab", "03-ptq-lab"))
    import src.quant as PTQQ
    import src.model as PTQM

    PTQM.fold_all_bn(model)

    n = 0
    with torch.no_grad():
        for _nm, mod in model.named_modules():
            if isinstance(mod, torch.nn.Conv2d):
                mod.weight.copy_(PTQQ.quantize_per_channel_sym(mod.weight.data, w_bits))
                n += 1

    print("权重量化  : fold BN + W%d 逐通道对称，覆盖 %d 个 Conv2d" % (w_bits, n))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=list(DATASET_SPECS))
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--opset", type=int, default=11,
                    help="ONNX opset。默认 11，ncnn 支持最成熟，不要随手改")
    ap.add_argument("--w-bits", type=int, default=0, choices=[0, 4, 8],
                    help="权重量化位宽。0=不量化（默认，保持阶段 1/2 的口径）；"
                         "4=fold BN 后逐通道对称量化到 4bit")
    ap.add_argument("--out", default=None, help="输出路径，默认 models/<arch>_<dataset>.onnx")
    args = ap.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    ckpt = ckpt_path(args.dataset, args.arch)
    if not os.path.isfile(ckpt):
        sys.exit("[缺] 权重不存在：%s" % ckpt)

    suffix = "_w%d" % args.w_bits if args.w_bits else ""
    out = args.out or os.path.join(
        MODEL_DIR, "%s_%s%s.onnx" % (args.arch, args.dataset, suffix))

    print("权重文件  : %s" % ckpt)
    print("输出      : %s" % out)
    print("opset     : %d" % args.opset)

    model = build_model(args.dataset, args.arch)
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    if args.w_bits:
        model = quantize_weights(model, args.w_bits)

    # 归一化参数只用于打印核对，不参与导出（模型吃归一化后的输入）
    spec = DATASET_SPECS[args.dataset]
    print("mean/std: %s / %s" % (spec["mean"], spec["std"]))

    dummy = torch.zeros(1, 3, 32, 32)
    with torch.no_grad():
        ref = model(dummy)

    torch.onnx.export(
        model, dummy, out,
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,   # 折叠常量，ncnn 侧图更干净
    )
    print("导出完成，参考输出 shape = %s" % (tuple(ref.shape),))

    # ---- 自检：ONNX 重新加载后和 PyTorch 对不上，后面全白测 ----
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as e:
        print("[提示] 跳过 ONNX 自检（缺 %s）。自检不是必须的，但建议装上："
              "pip install onnxruntime" % e.name)
        return

    m = onnx.load(out)
    onnx.checker.check_model(m)
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input": dummy.numpy()})[0]
    diff = float(abs(got - ref.numpy()).max())
    print("ONNX 自检 : max|diff| = %.3e  %s"
          % (diff, "PASS" if diff < 1e-4 else "FAIL"))
    if diff >= 1e-4:
        sys.exit("[FAIL] ONNX 与 PyTorch 输出不一致，先查导出再往下走")


if __name__ == "__main__":
    main()
