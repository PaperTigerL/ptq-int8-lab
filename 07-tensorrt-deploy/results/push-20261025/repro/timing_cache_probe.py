#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探针：TRT 11.4 的 python 绑定到底能不能读回 timing cache？

这条决定复现脚本能不能靠「固定 tactic cache」消除 engine 构建抖动
（07 README 说不能，这里实测确认一遍，并把结论落成可复跑的文件）。
"""
import tensorrt as trt

rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
print("tensorrt", trt.__version__)
print("hasattr(rt, 'deserialize_timing_cache'):",
      hasattr(rt, "deserialize_timing_cache"))
print("Runtime 里带 iming 的属性:",
      [x for x in dir(trt.Runtime) if "iming" in x])
print("模块级 TimingCache 相关:", [x for x in dir(trt) if "iming" in x])
