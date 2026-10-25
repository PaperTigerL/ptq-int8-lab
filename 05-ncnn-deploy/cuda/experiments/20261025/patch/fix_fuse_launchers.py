#!/usr/bin/env python3
"""Add the missing activation_params parameter to the two launcher definitions."""
import io

PATH = "/home/user/ncnn-cuda/src/layer/cuda/implicit_conv_cuda.cu"
PAIRS = [
("""                               bool requantize, int activation_type,
                               void* output, int out_cstep)""",
 """                               bool requantize, int activation_type,
                               const float* activation_params,
                               void* output, int out_cstep)"""),
("""                                      bool requantize, int activation_type,
                                      void* output, int out_cstep)""",
 """                                      bool requantize, int activation_type,
                                      const float* activation_params,
                                      void* output, int out_cstep)"""),
]

with io.open(PATH, "r", encoding="utf-8") as f:
    text = f.read()
for old, new in PAIRS:
    if old not in text:
        raise SystemExit("marker missing: %r" % old[:60])
    text = text.replace(old, new, 1)
with io.open(PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)
print("launcher signatures fixed")
