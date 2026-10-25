# -*- coding: utf-8 -*-
"""把「2x2 空间上的 3x3 pad=1 卷积」改写成等价的 block-GEMM（push-20261025）。

为什么等价：
  输入 2x2、pad=1、stride=1 时，每个输出像素的 3x3 窗口覆盖了整个 2x2 输入
  （只有 4 个真实 tap，另外 5 个是 padding 零）。也就是说
      y(i,j) = sum_{p,q} W[p-i+1, q-j+1] * x(p,q)      i,j,p,q in {0,1}
  这 4 个输出像素各自是 2x2 输入的线性组合，可拼成一个
      (Cout*4) x (Cin*4) 的稠密矩阵 M，Y = X_flat @ M^T
  原卷积算 4*9=36 个 MAC/通道对，块矩阵只算 16 个 -> 省 2.25x。
  权重内存 16 块 vs 9 个 tap（1.78x），对这个尺寸可忽略（每层 ~2 MB fp16）。

语义不变：M 的每个元素就是原卷积对应 tap 的权重，没有任何近似。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


def _shape(m, name):
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        if vi.name == name:
            return [x.dim_value if x.dim_value else 0 for x in vi.type.tensor_type.shape.dim]
    return None


def _attr(nd, name, default):
    for a in nd.attribute:
        if a.name == name:
            return list(a.ints) if a.ints else [a.i]
    return default


def blockgemm_2x2(model):
    m = shape_inference.infer_shapes(model, strict_mode=False)
    g = m.graph
    init = {i.name: i for i in g.initializer}
    changed = []
    replace = {}
    for nd in list(g.node):
        if nd.op_type != "Conv" or len(nd.input) < 2 or nd.input[1] not in init:
            continue
        shp = _shape(m, nd.input[0])
        oshp = _shape(m, nd.output[0])
        if not shp or not oshp or len(shp) != 4 or shp[2] != 2 or shp[3] != 2:
            continue
        if oshp[2] != 2 or oshp[3] != 2:
            continue
        if _attr(nd, "strides", [1, 1]) != [1, 1] or _attr(nd, "dilations", [1, 1]) != [1, 1]:
            continue
        if _attr(nd, "pads", [0, 0, 0, 0]) != [1, 1, 1, 1] or _attr(nd, "group", [1])[0] != 1:
            continue
        w = numpy_helper.to_array(init[nd.input[1]])
        if w.shape[2:] != (3, 3):
            continue
        cout, cin = w.shape[0], w.shape[1]
        dt = w.dtype
        # M[co*4 + i*2 + j, ci*4 + p*2 + q] = W[co, ci, p-i+1, q-j+1]
        M = np.zeros((cout * 4, cin * 4), dtype=np.float64)
        for i in range(2):
            for j in range(2):
                for p in range(2):
                    for q in range(2):
                        a, b = p - i + 1, q - j + 1
                        if 0 <= a < 3 and 0 <= b < 3:
                            rows = np.arange(cout) * 4 + i * 2 + j
                            cols = np.arange(cin) * 4 + p * 2 + q
                            # 必须用 np.ix_：M[rows][:, cols] = ... 写的是切片副本，静默失效
                            M[np.ix_(rows, cols)] = w[:, :, a, b].astype(np.float64)
        Wt = np.ascontiguousarray(M.T.astype(dt))          # (Cin*4, Cout*4) -> Y = X @ Wt
        nodes = []
        pre = nd.name.replace("/", "_") + "_bg"
        s1 = pre + "_shape_in"
        s2 = pre + "_shape_out"
        g.initializer.append(numpy_helper.from_array(np.array([-1, cin * 4], dtype=np.int64), s1))
        g.initializer.append(numpy_helper.from_array(np.array([-1, cout, 2, 2], dtype=np.int64), s2))
        wn = pre + "_W"
        g.initializer.append(numpy_helper.from_array(Wt, wn))
        nodes.append(helper.make_node("Reshape", [nd.input[0], s1], [pre + "_xflat"], name=pre + "_r1"))
        nodes.append(helper.make_node("MatMul", [pre + "_xflat", wn], [pre + "_yflat"], name=pre + "_mm"))
        last = pre + "_yflat"
        if len(nd.input) >= 3 and nd.input[2] in init:
            bias = numpy_helper.to_array(init[nd.input[2]]).astype(dt)
            bn = pre + "_bias"
            g.initializer.append(numpy_helper.from_array(
                np.repeat(bias, 4).reshape(1, cout * 4).astype(dt), bn))
            nodes.append(helper.make_node("Add", [last, bn], [pre + "_ybias"], name=pre + "_add"))
            last = pre + "_ybias"
        nodes.append(helper.make_node("Reshape", [last, s2], [pre + "_out"], name=pre + "_r2"))
        # 先记下来，后面统一按拓扑序重建节点表
        replace[nd.name] = dict(nodes=nodes, old=nd.output[0], new=pre + "_out")
        changed.append((nd.name, (cin, cout), float(w.nbytes), float(Wt.nbytes)))

    # 重建节点表：在 conv 原位置插入 Reshape/MatMul/Add/Reshape，并改接消费者
    remap = {v["old"]: v["new"] for v in replace.values()}
    new_nodes = []
    for nd in list(g.node):
        for k, inp in enumerate(nd.input):
            if inp in remap:
                nd.input[k] = remap[inp]
        if nd.name in replace:
            new_nodes.extend(replace[nd.name]["nodes"])
        else:
            new_nodes.append(nd)
    del g.node[:]
    g.node.extend(new_nodes)
    used = {i for nd in g.node for i in nd.input}
    for i in list(g.initializer):
        if i.name not in used:
            g.initializer.remove(i)
    onnx.checker.check_model(m)
    return m, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--verify-ort", action="store_true")
    ap.add_argument("--verify-n", type=int, default=200)
    args = ap.parse_args()

    m = onnx.load(args.src)
    out, changed = blockgemm_2x2(m)
    onnx.save(out, args.dst)
    print(f"{args.src} -> {args.dst}")
    print(f"  节点 {len(m.graph.node)} -> {len(out.graph.node)}；改写 {len(changed)} 个 2x2 卷积：")
    for n, s, wb, nb in changed:
        print(f"    {n}: {s[0]}->{s[1]} 3x3  权重 {wb/1e6:.2f} -> {nb/1e6:.2f} MB")
    print(f"  文件 {Path(args.dst).stat().st_size/1e6:.1f} MB（原 {Path(args.src).stat().st_size/1e6:.1f} MB）")

    if args.verify_ort:
        import onnxruntime as ort
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
        print(f"  ORT 校验 {args.verify_n} 张：logits 最大差 {d.max():.3e}，预测一致 "
              f"{int((p1 == p2).sum())}/{args.verify_n}，精度 {100*(p1==labels).mean():.2f}% / "
              f"{100*(p2==labels).mean():.2f}%")


if __name__ == "__main__":
    main()
