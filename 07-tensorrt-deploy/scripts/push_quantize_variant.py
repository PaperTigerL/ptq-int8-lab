# -*- coding: utf-8 -*-
"""对任意 fp32 ONNX 做 ORT 静态 PTQ（口径与 quantize_int8.py 完全一致），输出可指定。

quantize_int8.py 的路径是写死的；扫描/优化图要用不同底稿，所以拆出这一版，
量化语义直接复用同一个 CifarReader / strip_int32_bias_dq，保证可比。
"""
import argparse
from pathlib import Path
import sys

from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import load_raw
from quantize_int8 import CifarReader, strip_int32_bias_dq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--calib-n", type=int, default=1000)
    ap.add_argument("--calib-batch", type=int, default=50)
    ap.add_argument("--method", default="Entropy", choices=["Entropy", "MinMax", "Percentile", "Distribution"])
    args = ap.parse_args()

    raw = load_raw()
    total = raw.size // 3073
    calib_n = min(args.calib_n, total)
    raw_dst = Path(args.dst).with_name(Path(args.dst).stem + "_raw.onnx")
    print(f"校准 {calib_n} 张，method={args.method}, src={args.src}")
    quantize_static(
        model_input=args.src,
        model_output=str(raw_dst),
        calibration_data_reader=CifarReader(raw, calib_n, args.calib_batch),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        reduce_range=False,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        nodes_to_quantize=None,
        nodes_to_exclude=["/fc/Gemm"],
        use_external_data_format=False,
        calibrate_method=getattr(CalibrationMethod, args.method),
        calibration_providers=["CPUExecutionProvider"],
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )
    strip_int32_bias_dq(raw_dst, args.dst)
    print("final:", args.dst, Path(args.dst).stat().st_size, "bytes")


if __name__ == "__main__":
    main()
