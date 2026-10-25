"""PTQ → TensorRT int8（QDQ 图）。

为什么是 QDQ：
  TRT 11.4 的 python 绑定把 BuilderFlag.INT8 拿掉了，set_flag / cfg.flags 两条路都设不进去
  （实测读回永远是 64）。所以精度只能在图里表达——QuantizeLinear/DequantizeLinear 进去之后，
  TRT 自己挑 int8 kernel，不需要任何 builder flag。

量化语义（与 ptq-lab / ncnn 的差异，别混淆）：
  权重   per-channel + int8 对称   -> per_channel=True, weight_type=QInt8
  激活   per-tensor  + int8 对称   -> activation_type=QInt8, ActivationSymmetric=True
  校准   KL 散度                    -> CalibrationMethod.Entropy（ncnn2table 默认也是 KL）
  ⚠️ ncnn 那边是「uint8 非对称激活」，TRT 不支持：GPU 上只吃 zero_point 全 0 的对称量化
     （报 "Non-zero zero point is not supported"）。所以 QDQ 这一层必须改对称。

到 TRT 之前还要过两关（都是实测报错倒逼出来的）：
  1) fc.bias 会被量化成 int32 再 DequantizeLinear，TRT 的 DequantizeLinear 不收 int32
     -> nodes_to_exclude=["/fc/Gemm"]，这一层 512->10，算力占比可忽略
  2) Conv 的 bias 同理是 int32 -> strip_int32_bias_dq() 把这类 DQ 还原成浮点常量

校准集是 test_batch.raw 的前 --calib-n 张；评测必须 --skip 同样的张数，
否则是在自己标定过的样本上测精度。

用法:
    python quantize_int8.py                    # 1000 张 + Entropy
    python quantize_int8.py --method MinMax
"""
import argparse

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

from paths import TRT_WORKDIR, ensure_workdir, take_block, load_raw


class CifarReader(CalibrationDataReader):
    def __init__(self, raw, n, batch, name="input"):
        self.raw, self.n, self.batch, self.name = raw, n, batch, name
        self.i = 0

    def get_next(self):
        if self.i >= self.n:
            return None
        bs = min(self.batch, self.n - self.i)
        px, _ = take_block(self.raw, self.i, bs)
        self.i += bs
        return {self.name: px}


def strip_int32_bias_dq(src, dst):
    """把输入是 int32 的 DequantizeLinear 直接算成浮点常量。

    ORT 给 Conv 的 bias 生成 int32 + per-channel scale，TRT 的 DequantizeLinear 不收 int32
    （报 Error Code 4: must have type Int8/UInt8/...）。与其跟 TRT 的 importer 较劲，
    不如离线把它算开：scale 和 bias 同长，逐元素相乘即可。
    """
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
        try:
            val = (arr.astype(np.float32) - np.asarray(zp, dtype=np.float32)) * sc.astype(np.float32)
        except ValueError as e:
            print("  skip", nd.name, arr.shape, sc.shape, e)
            continue
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
    stale = [i for i in g.initializer if i.name not in used and i.name not in
             {n.name for n in new_inits}]
    for i in stale:
        g.initializer.remove(i)

    onnx.checker.check_model(m)
    onnx.save(m, str(dst))
    print(f"strip int32 bias DQ: 移除 {len(drop)} 节点 / 新增 {len(new_inits)} 浮点常量"
          f" / 清理 {len(stale)} 无用 initializer")
    kinds = {}
    for nd in m.graph.node:
        kinds[nd.op_type] = kinds.get(nd.op_type, 0) + 1
    print("节点类型:", kinds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-n", type=int, default=1000)
    ap.add_argument("--calib-batch", type=int, default=50)
    ap.add_argument("--method", default="Entropy",
                    choices=["Entropy", "MinMax", "Percentile", "Distribution"])
    args = ap.parse_args()

    ensure_workdir()
    src = TRT_WORKDIR / "resnet18_cifar10_fp32.onnx"
    raw_qdq = TRT_WORKDIR / "resnet18_cifar10_int8_raw.onnx"
    dst = TRT_WORKDIR / "resnet18_cifar10_int8.onnx"
    if not src.exists():
        raise SystemExit(f"先跑 export_onnx.py fp32，缺 {src}")

    raw = load_raw()
    total = raw.size // 3073
    calib_n = min(args.calib_n, total)
    print(f"校准样本 {calib_n}，method={args.method}")

    quantize_static(
        model_input=str(src),
        model_output=str(raw_qdq),
        calibration_data_reader=CifarReader(raw, calib_n, args.calib_batch),
        quant_format=QuantFormat.QDQ,
        per_channel=True,                    # 权重 per-channel
        reduce_range=False,
        activation_type=QuantType.QInt8,     # TRT 在 GPU 上只吃对称 int8
        weight_type=QuantType.QInt8,
        nodes_to_quantize=None,
        nodes_to_exclude=["/fc/Gemm"],       # 它的 bias 会被量化成 int32，TRT 不收
        use_external_data_format=False,
        calibrate_method=getattr(CalibrationMethod, args.method),
        calibration_providers=["CPUExecutionProvider"],
        extra_options={
            "ActivationSymmetric": True,     # 非零 zero_point 会被 TRT 直接拒绝
            "WeightSymmetric": True,
        },
    )
    print("raw QDQ:", raw_qdq.stat().st_size, "bytes")
    strip_int32_bias_dq(raw_qdq, dst)
    print("final:", dst.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
