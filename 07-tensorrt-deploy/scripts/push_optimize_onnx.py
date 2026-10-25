# -*- coding: utf-8 -*-
"""语义等价的 ONNX 图优化（push-20261025）。

核心一条：**1x1 空间上的 3x3 卷积，只有中心 tap 会碰到真实像素**。

  layer4 的输入在 32x32 CIFAR 输入下已经缩到 1x1，
  再叠 3x3 pad=1 的卷积，输出 1x1 —— 3x3 窗口里 8 个 tap 全是 padding 零，
  算力白扔 9 倍（profile 实测这三层占 fp16 总时的 ~16%）。
  把权重切成中心 tap、kernel 改成 1x1、pad 清零，数学上完全等价：
      y = sum_{di,dj} W[di,dj] * x_pad[di,dj]   （x_pad 只有中心非零）
        = W[1,1] * x
  不改语义、不改权重数值，只改表达。

同时做一件小事：把 Conv 的 kernel 形状之外的属性（strides/dilations/group）
原样保留；权重初始值按 [out,in,1,1] 切片。

用法:
    python push_optimize_onnx.py --in work/resnet18_cifar10_fp16.onnx \
        --out work/resnet18_cifar10_fp16_c1x1.onnx
    python push_optimize_onnx.py --in ... --out ... --revert-check
"""
import argparse
import numpy as np
import onnx
from onnx import numpy_helper


def _shape(m, name):
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        if vi.name == name:
            d = [x.dim_value for x in vi.type.tensor_type.shape.dim]
            return d
    return None


def collapse_1x1_convs(model):
    """把输入/输出都是 1x1 空间、奇数 kernel、stride=1、pad=(k-1)/2 的 Conv 改成 1x1。"""
    import onnx.shape_inference
    m = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    g = m.graph
    init = {i.name: i for i in g.initializer}
    changed, skipped = [], []
    for nd in g.node:
        if nd.op_type != "Conv" or len(nd.input) < 2 or nd.input[1] not in init:
            continue
        xin = _shape(m, nd.input[0])
        xout = _shape(m, nd.output[0])
        if not xin or not xout or len(xin) != 4 or len(xout) != 4:
            continue
        if xin[2] != 1 or xin[3] != 1 or xout[2] != 1 or xout[3] != 1:
            continue
        w = numpy_helper.to_array(init[nd.input[1]])
        if w.ndim != 4:
            continue
        k = w.shape[2]
        if k == 1 or w.shape[3] != k or k % 2 == 0:
            skipped.append((nd.name, w.shape))
            continue
        # strides / dilations 必须为 1，pad 必须恰好 (k-1)/2（两侧相等）
        def _attr(name, default):
            for a in nd.attribute:
                if a.name == name:
                    return list(a.ints) if a.ints else [a.i]
            return default
        strides = _attr("strides", [1, 1])
        dils = _attr("dilations", [1, 1])
        pads = _attr("pads", [0, 0, 0, 0])
        groups = _attr("group", 1)[0]
        c = k // 2
        if strides != [1, 1] or dils != [1, 1] or pads != [c, c, c, c]:
            skipped.append((nd.name, dict(w=w.shape, s=strides, d=dils, p=pads)))
            continue
        w1 = np.ascontiguousarray(w[:, :, c, c].reshape(w.shape[0], w.shape[1], 1, 1))
        new_name = nd.input[1] + "_c1x1"
        new_init = numpy_helper.from_array(w1, new_name)
        g.initializer.append(new_init)
        init[new_name] = new_init
        nd.input[1] = new_name
        # 改写属性：kernel_shape/pads；strides/dilations/group 保持
        keep = [a for a in nd.attribute if a.name not in ("kernel_shape", "pads")]
        nd.attribute.clear()
        for a in keep:
            nd.attribute.append(a)
        nd.attribute.append(onnx.helper.make_attribute("kernel_shape", [1, 1]))
        nd.attribute.append(onnx.helper.make_attribute("pads", [0, 0, 0, 0]))
        changed.append((nd.name, tuple(w.shape)))
    # 清理不再被引用的旧权重
    used = {i for nd in g.node for i in nd.input}
    for i in list(g.initializer):
        if i.name not in used:
            g.initializer.remove(i)
    onnx.checker.check_model(m)
    return m, changed, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--verify-ort", action="store_true",
                    help="用 ORT CPU 跑 200 张真实图片，对比原图/优化图的 logits 与预测")
    ap.add_argument("--verify-n", type=int, default=200)
    args = ap.parse_args()

    m = onnx.load(args.src)
    out, changed, skipped = collapse_1x1_convs(m)
    onnx.save(out, args.dst)
    print(f"{args.src} -> {args.dst}")
    print(f"  节点 {len(m.graph.node)} -> {len(out.graph.node)}；折叠 {len(changed)} 个 1x1 空间卷积：")
    for n, s in changed:
        print(f"    {n}: {s} -> ({s[0]},{s[1]},1,1)")
    if skipped:
        print("  跳过：", skipped[:6])
    print(f"  文件 {onnx.load(args.dst).ByteSize()/1e6:.1f} MB（原 {onnx.load(args.src).ByteSize()/1e6:.1f} MB）")

    if args.verify_ort:
        import onnxruntime as ort
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from paths import load_raw, take_block
        raw = load_raw()
        px, labels = take_block(raw, 1000, args.verify_n)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        s1 = ort.InferenceSession(args.src, so, providers=["CPUExecutionProvider"])
        s2 = ort.InferenceSession(args.dst, so, providers=["CPUExecutionProvider"])
        y1 = s1.run(None, {s1.get_inputs()[0].name: px})[0]
        y2 = s2.run(None, {s2.get_inputs()[0].name: px})[0]
        d = np.abs(y1.astype(np.float64) - y2.astype(np.float64))
        p1, p2 = y1.argmax(1), y2.argmax(1)
        print(f"  ORT 校验 {args.verify_n} 张：logits 最大差 {d.max():.3e}，"
              f"平均差 {d.mean():.3e}，预测一致 {int((p1 == p2).sum())}/{args.verify_n}，"
              f"原图精度 {100*(p1==labels).mean():.2f}% / 优化后 {100*(p2==labels).mean():.2f}%")


if __name__ == "__main__":
    main()
