# -*- coding: utf-8 -*-
"""生成 int8/fp16 混跑图：在 fp16 ONNX 上做 QDQ 量化。

动机：TRT 11.4 没有 BuilderFlag.FP16/INT8，精度只能写在图里。
- 纯 int8（quantize_int8.py）：未量化的算子回落 **fp32**，这些层反而更慢
- 混跑：整张图是 fp16，只把选中的 Conv 换成 int8 QDQ -> 其余算子保持 fp16

于是「哪些层走 int8」变成一个图级选择，可以真做 A/B：
  --exclude-layer4   layer4 保持 fp16，stem/layer1-3 走 int8
  （默认全量化）     全部 Conv 走 int8，其余 fp16

注意 fp16 图里 ORT 会把 Conv bias 量化成 int32，TRT 不收，仍要 strip；
这里按图自身 dtype 输出常量（fp16 图 -> fp16 常量）。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import load_raw, take_block


class F16Reader(CalibrationDataReader):
    def __init__(self, raw, n, batch, name="input"):
        self.raw, self.n, self.batch, self.name, self.i = raw, n, batch, name, 0

    def get_next(self):
        if self.i >= self.n:
            return None
        bs = min(self.batch, self.n - self.i)
        px, _ = take_block(self.raw, self.i, bs)
        self.i += bs
        return {self.name: px.astype(np.float16)}


def strip_int32_bias_dq(src, dst, out_dtype):
    m = onnx.load(str(src))
    g = m.graph
    init = {i.name: i for i in g.initializer}
    new_inits, drop = [], []
    for nd in list(g.node):
        if nd.op_type != "DequantizeLinear" or len(nd.input) < 2:
            continue
        x, s = nd.input[0], nd.input[1]
        if x not in init or s not in init:
            continue
        arr = numpy_helper.to_array(init[x])
        if arr.dtype != np.int32:
            continue
        sc = numpy_helper.to_array(init[s])
        zp = (numpy_helper.to_array(init[nd.input[2]])
              if len(nd.input) > 2 and nd.input[2] in init else 0)
        val = ((arr.astype(np.float64) - np.asarray(zp, dtype=np.float64))
               * sc.astype(np.float64)).astype(out_dtype)
        name = x + "_f32"
        new_inits.append(numpy_helper.from_array(np.ascontiguousarray(val), name))
        for consumer in g.node:
            for k, inp in enumerate(consumer.input):
                if inp == nd.output[0]:
                    consumer.input[k] = name
        drop.append(nd)
    for nd in drop:
        g.node.remove(nd)
    used = {inp for nd in g.node for inp in nd.input} | {o.name for o in g.output}
    g.initializer.extend(new_inits)
    stale = [i for i in g.initializer if i.name not in used
             and i.name not in {n.name for n in new_inits}]
    for i in stale:
        g.initializer.remove(i)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))
    print(f"strip int32 bias DQ: 移除 {len(drop)} / 新增 {len(new_inits)} 个 {np.dtype(out_dtype).name} 常量")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="fp16 ONNX")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--calib-n", type=int, default=1000)
    ap.add_argument("--method", default="Entropy", choices=["Entropy", "MinMax"])
    ap.add_argument("--exclude-layer4", action="store_true")
    ap.add_argument("--only-layer4", action="store_true")
    args = ap.parse_args()

    raw = load_raw()
    excl = ["/fc/Gemm"]
    quant_only = None
    if args.exclude_layer4:
        excl += [n for n in []]
    m0 = onnx.load(args.src)
    convs = [n.name for n in m0.graph.node if n.op_type == "Conv"]
    if args.exclude_layer4:
        excl += [n for n in convs if "/layer4/" in n]
    if args.only_layer4:
        quant_only = [n for n in convs if "/layer4/" in n]
    print("量化排除:", excl if len(excl) < 8 else f"{excl[:3]}...({len(excl)})")
    if quant_only:
        print("只量化:", quant_only)

    raw_dst = str(Path(args.dst).with_name(Path(args.dst).stem + "_raw.onnx"))
    quantize_static(
        model_input=args.src, model_output=raw_dst,
        calibration_data_reader=F16Reader(raw, min(args.calib_n, 10000), 50),
        quant_format=QuantFormat.QDQ, per_channel=True, reduce_range=False,
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        nodes_to_quantize=quant_only,
        nodes_to_exclude=excl if quant_only is None else ["/fc/Gemm"],
        calibrate_method=getattr(CalibrationMethod, args.method),
        calibration_providers=["CPUExecutionProvider"],
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )
    strip_int32_bias_dq(raw_dst, args.dst, np.float16)
    import collections
    m = onnx.load(args.dst)
    print("ops", dict(collections.Counter(n.op_type for n in m.graph.node)))
    print("final:", args.dst, Path(args.dst).stat().st_size, "bytes")


if __name__ == "__main__":
    main()
