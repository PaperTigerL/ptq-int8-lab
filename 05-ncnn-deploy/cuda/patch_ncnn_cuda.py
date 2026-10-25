"""给 ncnn-with-cuda 打补丁。

原因：ncnn 为了移动端体积整体开了 -fno-exceptions，但这个分支在两处写了 throw，
导致 GCC 直接报 "exception handling disabled"。
-CMAKE_CXX_FLAGS=-fexceptions 救不了，因为 target 级的 -fno-exceptions 会覆盖它。

处理：把 throw 换成返回错误码。这两条路径我们的模型都到不了：
  - innerproduct_cuda.cpp 的 throw 条件是 bottom_blob.c * num_output >= 65000，
    我们的 fc 是 512*10 = 5120
  - padding_cuda.cpp 的 int8 路径 ResNet-18 不用（pad 在 Convolution 层内部做）
所以改掉它们不改变任何实际行为，也不给整个库引入异常。
"""
from pathlib import Path

ROOT = Path("/home/user/ncnn-cuda/src/layer/cuda")
TARGETS = ["innerproduct_cuda.cpp", "padding_cuda.cpp"]
OLD = 'throw  std::runtime_error("Not implemented");'
NEW = 'return -1; // patched: 这条 int8 路径未实现（见 tools/patch_ncnn_cuda.py）'

for name in TARGETS:
    p = ROOT / name
    txt = p.read_text()
    cnt = txt.count(OLD)
    if cnt == 0:
        print(f"{name}: 没有找到目标语句（可能已打过补丁）")
        continue
    p.write_text(txt.replace(OLD, NEW))
    print(f"{name}: 替换 {cnt} 处")
