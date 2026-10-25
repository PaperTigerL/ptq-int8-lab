# -*- coding: utf-8 -*-
"""push-20261025b · 位置分解：把「padding 吃掉大部分 tap」的 3x3 卷积折成位置专属子卷积。

背景（见 results/push-20261025/README.md §2/§9）：
  layer4 在 32x32 输入下空间已经是 1x1，3x3 pad=1 卷积有 8/9 的乘零 —— 那条已被
  push_optimize_onnx.py 折成 1x1（int8 0.0189 -> 0.00744）。剩下的大头是同一种浪费的
  另外两个形状：

    layer3  输入 2x2、3x3 pad=1、stride 1、输出 2x2  -> 36 个 tap 里只有 16 个真实（2.25x 浪费）
    layer4.0/conv1  输入 2x2、3x3 pad=1、stride 2、输出 1x1 -> 9 个 tap 里只有 4 个真实

本脚本处理的是一类**可证恒等**的形状族：**输入空间每个维都不大于 kernel、pad=(k-1)/2**。
这时每个输出位置的有效 tap 都是 min(k,H) 个，并且**所有输出位置读的都是同一块完整输入**
（区别只在权重切到 3x3 里的哪几行/几列）。于是：

  A) stride=1、输出 HxW（layer3 的 2x2）：
     一次 Conv(kernel = HxW = 2x2, out_channels = (H*W)*Cout, pads=0, stride=1)
     产出 [N, 4*Cout, 1, 1]，再用 DepthToSpace(blocksize=H, mode=DCR) 还原成 [N, Cout, 2, 2]。
     （DCR 的通道映射：y[n,c,i,j] = x[n, (i*bs+j)*Cout + c]，与下面「块主序」的权重排布一致；
       已用 work/scratch/d2s_probe.py 在 ORT 上实测确认，不靠文档。）
     恒等式：y[i,j,c] = sum_{u,v} W[c, :, p-i+u, p-j+v] * x[:, u, v]
             = sum_{a,b} W[c, :, a, b] * x_pad[:, i+a-p, j+b-p]   （x_pad 只有真实区非零）
     —— 就是原卷积；只是把「每个输出位置用哪 4 个 tap」写成了一张一次算完的权重矩阵。
     MAC：2.36M -> 1.05M（2.25x）；kernel 数不变（1 conv + 1 shuffle）。

  B) stride=2、输出 1x1（layer4.0/conv1 的 2x2->1x1）：
     只有一个输出位置，直接用 W[:, :, 1:3, 1:3] 当 2x2 kernel，pads=0、stride=1。
     单 kernel，MAC 2.25x 下降。

不是这个形状族的卷积一律不动（layer2 的 4x4 输入上，输出位置有 3 种不同的 tap 形状、
9 种不同的输入窗口，一个 kernel 表达不了 —— 需要 9 个 kernel，见 README）。

用法:
    python push_decompose.py --in work/resnet18_cifar10_fp32.onnx \
        --out results/push-20261025b/resnet18_cifar10_fp32_decomp.onnx --verify-ort
    # 或只折某一族：--only layer3 / --only strand / --only layer3,layer4s2
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, helper


# ---------------------------------------------------------------- 形状工具

def dim_plan(H, k, p, s):
    """每个输出位置的有效 kernel 切片与输入窗口。

    返回 (OH, taps)：taps[i] = (a_lo, a_hi, r_lo, r_hi)，a 是 kernel 下标，r 是输入下标。
    只用一维，二维取笛卡尔积。
    """
    OH = (H + 2 * p - k) // s + 1
    taps = []
    for i in range(OH):
        lo = max(0, p - s * i)                 # i*s - p + a >= 0
        hi = min(k - 1, H - 1 + p - s * i)     # i*s - p + a <= H-1
        if lo > hi:
            return OH, None
        taps.append((lo, hi, s * i - p + lo, s * i - p + hi))
    return OH, taps


def uniform_dim(H, k, p, s):
    """该维是否满足「每个输出位置读同一块完整输入、tap 数相同」。

    满足则返回 (OH, t, los)：t = 每个位置的 tap 数，los[i] = 位置 i 的 kernel 切起点。
    """
    OH, taps = dim_plan(H, k, p, s)
    if not taps:
        return None
    t = taps[0][1] - taps[0][0] + 1
    if any(hi - lo + 1 != t for lo, hi, _, _ in taps):
        return None
    if any((r_lo, r_hi) != (0, H - 1) for _, _, r_lo, r_hi in taps):
        return None
    return OH, t, [lo for lo, _, _, _ in taps]


def _vi_shape(m, name):
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        if vi.name == name:
            return [d.dim_value for d in vi.type.tensor_type.shape.dim]
    return None


def _attr(nd, name, default):
    for a in nd.attribute:
        if a.name == name:
            return list(a.ints) if a.ints else [a.i]
    return default


# ---------------------------------------------------------------- 主变换

def decompose(model, only=None, verbose=True):
    m = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    g = m.graph
    init = {i.name: i for i in g.initializer}
    changed, skipped = [], []

    for nd in g.node:
        if nd.op_type != "Conv" or len(nd.input) < 2 or nd.input[1] not in init:
            continue
        xin, xout = _vi_shape(m, nd.input[0]), _vi_shape(m, nd.output[0])
        if not xin or not xout or len(xin) != 4 or len(xout) != 4:
            continue
        w = numpy_helper.to_array(init[nd.input[1]])
        if w.ndim != 4:
            continue
        kh, kw = w.shape[2], w.shape[3]
        strides = _attr(nd, "strides", [1, 1])
        pads = _attr(nd, "pads", [0, 0, 0, 0])
        dils = _attr(nd, "dilations", [1, 1])
        groups = _attr(nd, "group", 1)[0]
        if groups != 1 or dils != [1, 1] or kh != kw or kh % 2 == 0:
            continue
        k = kh
        p = (k - 1) // 2
        if pads != [p, p, p, p]:
            continue
        H, W = xin[2], xin[3]
        OH, OW = xout[2], xout[3]
        if H > k or W > k:
            continue                      # 只有「输入不超过 kernel」才是统一形状族
        sh, sw = strides[0], strides[1]
        ph = uniform_dim(H, k, p, sh)
        pw = uniform_dim(W, k, p, sw)
        if ph is None or pw is None:
            skipped.append((nd.name, f"非统一形状 H{H} W{W} k{k} s{sh}"))
            continue
        oh, th, los_h = ph
        ow, tw, los_w = pw
        if (oh, ow) != (OH, OW) or OH * OW < 1:
            skipped.append((nd.name, f"输出不符 {oh}x{ow} vs {OH}x{OW}"))
            continue
        tag = "layer3" if (sh == 1 and OH > 1) else ("strand" if OH == 1 else "other")
        if only and tag not in only:
            continue

        if (th, tw) == (k, k) and pads == [0, 0, 0, 0]:
            skipped.append((nd.name, "已是等价形状，不动"))
            continue
        cout, cin = w.shape[0], w.shape[1]
        # 新权重：[OH*OW*cout, cin, th, tw]，通道块顺序 = 位置行主序（与 DepthToSpace DCR 的映射一致）
        nw = np.empty((OH * OW * cout, cin, th, tw), dtype=w.dtype)
        for i in range(OH):
            for j in range(OW):
                blk = (i * OW + j) * cout
                nw[blk:blk + cout] = w[:, :, los_h[i]:los_h[i] + th, los_w[j]:los_w[j] + tw]
        nw = np.ascontiguousarray(nw)
        # bias：每个位置块用同一份 bias，按块主序拼 OH*OW 份（缺了它 ORT 会直接报错）
        if len(nd.input) > 2 and nd.input[2]:
            b = numpy_helper.to_array(init[nd.input[2]])
            if b.shape[0] != cout * OH * OW:
                bw = nd.input[2] + "_decomp"
                g.initializer.append(numpy_helper.from_array(
                    np.ascontiguousarray(np.tile(b, OH * OW)), bw))
                init[bw] = bw
                nd.input[2] = bw

        conv_name = nd.name
        out_name = nd.output[0]
        mid = f"{conv_name}/decomp_conv"
        wname = nd.input[1] + "_decomp"
        g.initializer.append(numpy_helper.from_array(nw, wname))
        init[wname] = wname
        nd.input[1] = wname
        nd.output[0] = mid
        nd.name = f"{conv_name}/decomp_conv"
        keep = [a for a in nd.attribute if a.name not in ("kernel_shape", "pads", "strides")]
        nd.attribute.clear()
        for a in keep:
            nd.attribute.append(a)
        nd.attribute.append(helper.make_attribute("kernel_shape", [th, tw]))
        nd.attribute.append(helper.make_attribute("strides", [1, 1]))
        nd.attribute.append(helper.make_attribute("pads", [0, 0, 0, 0]))

        if OH > 1 or OW > 1:
            if OH != OW:
                raise SystemExit(f"{conv_name}: 非方形位置网格 {OH}x{OW}，DepthToSpace 不支持")
            d2s = helper.make_node("DepthToSpace", [mid], [out_name],
                                   name=f"{conv_name}/decomp_d2s", blocksize=OH, mode="DCR")
            idx = list(g.node).index(nd)
            g.node.insert(idx + 1, d2s)
        else:
            nd.output[0] = out_name
        changed.append(dict(name=conv_name, tag=tag, w=tuple(int(x) for x in w.shape),
                            nw=tuple(int(x) for x in nw.shape), in_hw=(H, W), out_hw=(OH, OW),
                            s=(sh, sw), mac_old=int(k * k * cin * cout * OH * OW),
                            mac_new=int(th * tw * cin * cout * OH * OW)))

    used = {i for nd in g.node for i in nd.input}
    for i in list(g.initializer):
        if i.name not in used:
            g.initializer.remove(i)
    # 清掉旧的 value_info（形状已变），交给下游重新推断
    onnx.checker.check_model(m)
    return m, changed, skipped


# ---------------------------------------------------------------- 校验

def verify_ort(src, dst, n=200, tag=""):
    import onnxruntime as ort
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from paths import load_raw, take_block
    raw = load_raw()
    px, labels = take_block(raw, 1000, n)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    outs = []
    for path in (src, dst):
        s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        inp = s.get_inputs()[0]
        feed = px.astype(np.float16 if "float16" in inp.type else np.float32)
        outs.append(s.run(None, {inp.name: feed})[0].astype(np.float64))
    y1, y2 = outs
    d = np.abs(y1 - y2)
    p1, p2 = y1.argmax(1), y2.argmax(1)
    print(f"  ORT 校验{n} 张：logits max|d| {d.max():.3e} mean|d| {d.mean():.3e}，"
          f"预测一致 {int((p1 == p2).sum())}/{n}，精度 {100 * (p1 == labels).mean():.2f}% -> "
          f"{100 * (p2 == labels).mean():.2f}%")
    return dict(max_diff=float(d.max()), mean_diff=float(d.mean()),
                agree=int((p1 == p2).sum()), n=n,
                acc_src=float(100 * (p1 == labels).mean()), acc_dst=float(100 * (p2 == labels).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--only", default="", help="逗号分隔：layer3,strand（默认全做）")
    ap.add_argument("--verify-ort", action="store_true")
    ap.add_argument("--verify-n", type=int, default=200)
    args = ap.parse_args()

    only = [x for x in args.only.split(",") if x] or None
    model = onnx.load(args.src)
    out, changed, skipped = decompose(model, only=only)
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(out, args.dst)

    print(f"{args.src} -> {args.dst}")
    print(f"  节点 {len(model.graph.node)} -> {len(out.graph.node)}，改写 {len(changed)} 个卷积：")
    for c in changed:
        print(f"    [{c['tag']:6s}] {c['name']:44s} W {c['w']} -> {c['nw']}  "
              f"MAC {c['mac_old']} -> {c['mac_new']} "
              f"({c['mac_old']/max(1,c['mac_new']):.2f}x)")
    if skipped:
        print("  跳过：", skipped[:8])
    print(f"  文件 {Path(args.dst).stat().st_size/1e6:.1f} MB")
    if args.verify_ort:
        r = verify_ort(args.src, args.dst, args.verify_n)
        outdir = Path(args.dst).parent
        (outdir / "ort_verify.json").write_text(__import__("json").dumps(
            dict(src=args.src, dst=args.dst, **r), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()