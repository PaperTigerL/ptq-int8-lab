# -*- coding: utf-8 -*-
"""PTQC 导出/读回单测（纯标准库，不依赖 torch）。

用法：
    python tests/test_export_ptqc.py
    pytest tests/test_export_ptqc.py

ptqc 是纯格式模块，只依赖 struct/math。这里**直接按文件路径加载** ptqc.py，
绕开 `ptq_toolkit.export.__init__`（它 import 了依赖 torch 的 exporter）——
这样在无 torch 的环境也能验证格式层，与 06 load_calib 的字节布局对齐。
"""
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "ptqc", _ROOT / "ptq_toolkit" / "export" / "ptqc.py")
ptqc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ptqc)
write_ptqc, read_ptqc = ptqc.write_ptqc, ptqc.read_ptqc


def _tmp(name="t.ptqc"):
    d = tempfile.mkdtemp()
    return Path(d) / name


def _f32(x: float) -> float:
    """值经 float32 存储/读回后的表示（PTQC 存的就是 float32）。"""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def test_roundtrip_layers():
    for ranges in (
        [],                                   # 0 层
        [(0.0, 1.0)],                         # 1 层
        [(-3.5, 2.25), (0.0, 6.0), (-1.0, 1.0), (0.125, 7.5)],  # 多层（均可被 float32 精确表示）
    ):
        p = _tmp()
        info = write_ptqc(p, ranges)
        assert info["n"] == len(ranges)
        assert info["bytes"] == 8 + 8 * len(ranges)
        got = read_ptqc(p)
        assert got == ranges, f"{ranges} -> {got}"
    print("test_roundtrip_layers OK")


def test_roundtrip_float32_precision():
    # 非 float32 精确值：读回应等于「按 float32 存储后的表示」，逐值一致到存储精度。
    ranges = [(0.1, 0.3), (-0.7, 123.456), (1e-8, 1e8)]
    expect = [(_f32(lo), _f32(hi)) for lo, hi in ranges]
    p = _tmp()
    write_ptqc(p, ranges)
    assert read_ptqc(p) == expect
    print("test_roundtrip_float32_precision OK")


def test_exact_file_size():
    for n in (0, 1, 3, 20):
        ranges = [(float(-i), float(i)) for i in range(n)]
        p = _tmp()
        write_ptqc(p, ranges)
        assert p.stat().st_size == 8 + 8 * n, p.stat().st_size
    print("test_exact_file_size OK")


def test_header_layout():
    p = _tmp()
    write_ptqc(p, [(1.5, 2.5), (-4.0, -3.0)])
    data = p.read_bytes()
    assert data[:4] == b"PTQC"
    assert struct.unpack_from("<i", data, 4)[0] == 2
    lo, hi = struct.unpack_from("<2f", data, 8)
    assert (lo, hi) == (1.5, 2.5)
    lo, hi = struct.unpack_from("<2f", data, 16)
    assert (lo, hi) == (-4.0, -3.0)
    print("test_header_layout OK")


def test_reject_invalid_values():
    p = _tmp()
    for bad in ([(float("nan"), 1.0)], [(0.0, float("inf"))],
                [(float("-inf"), 0.0)]):
        try:
            write_ptqc(p, bad)
            raise AssertionError(f"应拒绝非有限值 {bad}")
        except ValueError:
            pass
    try:
        write_ptqc(p, [(2.0, 1.0)])          # lo > hi
        raise AssertionError("应拒绝 lo>hi")
    except ValueError:
        pass
    try:
        write_ptqc(p, [(0.0, 1.0, 2.0)])     # 非两元
        raise AssertionError("应拒绝非 (lo,hi)")
    except ValueError:
        pass
    print("test_reject_invalid_values OK")


def test_reject_negative_n():
    p = _tmp()
    try:
        write_ptqc(p, [], n=-1)              # 显式负 n
        raise AssertionError("应拒绝 n<0")
    except ValueError:
        pass
    try:
        write_ptqc(p, [(0.0, 1.0)], n=5)     # n 与 ranges 不符
        raise AssertionError("应拒绝 n!=len(ranges)")
    except ValueError:
        pass
    print("test_reject_negative_n OK")


def test_read_rejects_corrupt():
    p = _tmp()
    # magic 不符
    p.write_bytes(b"XXXX" + struct.pack("<i", 0))
    try:
        read_ptqc(p); raise AssertionError("应拒绝坏 magic")
    except ValueError:
        pass
    # 头里 n 为负
    p.write_bytes(b"PTQC" + struct.pack("<i", -3))
    try:
        read_ptqc(p); raise AssertionError("应拒绝负 n")
    except ValueError:
        pass
    # 大小与 n 不符（声明 2 层只写 1 层）
    p.write_bytes(b"PTQC" + struct.pack("<i", 2) + struct.pack("<2f", 0.0, 1.0))
    try:
        read_ptqc(p); raise AssertionError("应拒绝大小不符")
    except ValueError:
        pass
    print("test_read_rejects_corrupt OK")


class _Stat:
    """最小化的激活统计双元组（模拟 torch tensor 的 .min()/.max()）。"""

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def min(self):
        return self.lo

    def max(self):
        return self.hi


def test_ranges_follow_conv_order():
    names = ["conv1", "layer1.0.conv1", "layer1.0.downsample.0"]
    store = {
        "layer1.0.downsample.0": _Stat(0.25, 0.75),
        "conv1": _Stat(0.0, 1.0),
        "layer1.0.conv1": _Stat(-2.5, 3.5),
    }
    got = ptqc.ranges_from_store(store, names)
    assert got == [(0.0, 1.0), (-2.5, 3.5), (0.25, 0.75)], got
    print("test_ranges_follow_conv_order OK")


def test_ranges_reject_missing_or_extra():
    try:
        ptqc.ranges_from_store({"a": _Stat(0.0, 1.0)}, ["a", "b"])
        raise AssertionError("缺层应报错")
    except ValueError:
        pass
    try:
        ptqc.ranges_from_store(
            {"a": _Stat(0.0, 1.0), "fc": _Stat(0.0, 1.0)}, ["a"])
        raise AssertionError("多余非 conv 键应报错")
    except ValueError:
        pass
    print("test_ranges_reject_missing_or_extra OK")


def _ptqw_r18():
    return _ROOT / "results" / "resnet18_folded_w8a8.bin"


def _calib06():
    # Windows 直跑与 WSL 跑都用同一份 06 夹具；两种路径都探测，
    # 否则 WSL 下 "D:/..." 会被当成相对路径，测试假失败。
    for p in (Path("D:/WorkBuddy/ShowBro/lab/06-gpu-int8/cpp/calib_minmax.bin"),
              Path("/mnt/d/WorkBuddy/ShowBro/lab/06-gpu-int8/cpp/calib_minmax.bin")):
        if p.is_file():
            return p
    return Path("D:/WorkBuddy/ShowBro/lab/06-gpu-int8/cpp/calib_minmax.bin")


def test_weight_conv_count_real():
    p = _ptqw_r18()
    if not p.is_file():
        print("test_weight_conv_count_real SKIP（缺 resnet18 PTQW）")
        return
    magic, n = ptqc.weight_conv_count(p)
    assert magic == "PTQW" and n == 20, (magic, n)
    p2 = _ROOT / "results" / "mobilenet_v2_folded_w8a8.bin"
    if p2.is_file():
        magic2, n2 = ptqc.weight_conv_count(p2)
        assert magic2 == "PTQ2" and n2 == 35, (magic2, n2)
    print("test_weight_conv_count_real OK")


def test_layer_count_matches_ptqw_cli():
    p = _ptqw_r18()
    if not p.is_file():
        print("test_layer_count_matches_ptqw_cli SKIP（缺 resnet18 PTQW）")
        return
    src = _calib06()
    assert src.is_file(), f"缺 06 真实 PTQC 夹具：{src}"
    ranges = ptqc.read_ptqc(src)
    assert len(ranges) == 20, f"06 calib 应为 20 层，实得 {len(ranges)}"
    rj = _tmp("ranges20.json")
    rj.write_text(json.dumps(ranges), encoding="utf-8")
    out = _tmp("cli_r18.ptqc")
    r = subprocess.run(
        [sys.executable, "-m", "ptq_toolkit", "ptqc",
         "--ranges", str(rj), "--out", str(out), "--weights", str(p)],
        cwd=str(_ROOT), capture_output=True)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert len(ptqc.read_ptqc(out)) == 20
    assert ptqc.read_ptqc(out) == ranges
    print("test_layer_count_matches_ptqw_cli OK")


def test_cli_rejects_count_mismatch():
    p = _ptqw_r18()
    if not p.is_file():
        print("test_cli_rejects_count_mismatch SKIP（缺 resnet18 PTQW）")
        return
    rj = _tmp("ranges2.json")
    rj.write_text(json.dumps([[0.0, 1.0], [0.0, 2.0]]), encoding="utf-8")
    out = _tmp("bad.ptqc")
    r = subprocess.run(
        [sys.executable, "-m", "ptq_toolkit", "ptqc",
         "--ranges", str(rj), "--out", str(out), "--weights", str(p)],
        cwd=str(_ROOT), capture_output=True)
    assert r.returncode != 0, "层数 2 != PTQW 20，CLI 应失败"
    print("test_cli_rejects_count_mismatch OK")


def test_bytes_match_06_calib():
    src = _calib06()
    if not src.is_file():
        print("test_bytes_match_06_calib SKIP（缺 06 calib_minmax.bin）")
        return
    out = _tmp("reemit.ptqc")
    ptqc.write_ptqc(out, ptqc.read_ptqc(src))
    assert out.read_bytes() == src.read_bytes(), "重写字节应与 06 产物完全一致"
    print("test_bytes_match_06_calib OK")


def main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\n全部通过（{len(tests)} 项）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
