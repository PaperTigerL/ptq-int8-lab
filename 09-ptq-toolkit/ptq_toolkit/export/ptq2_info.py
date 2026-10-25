# -*- coding: utf-8 -*-
"""PTQ2 文件的结构化读取（纯标准库，不依赖 torch）。

用途：导出后立刻校验「写出的字节确实能被按同一布局读回」，
而不是只信 exporter 的返回值。校验项：
  · magic / header 合法性
  · 每个 conv 的元信息（含 groups、relu 0/1/2）
  · 每个 block 的尾部 -1 补位、ds/stride/residual
  · fc 维度
  · **解析终点必须恰好等于文件长度**（截断/尾部多余都拒绝）

与 06 的 `ptq2_reader.h` / `ptq_model.h` 是同一套布局；
两边的严格校验共同锁死格式，避免「Python 写对、C++ 读错」只在一侧暴露。
"""
import struct
from pathlib import Path

MAGIC = b"PTQ2"


def _need(data: bytes, off: int, n: int, what: str) -> None:
    if off + n > len(data):
        raise ValueError(
            f"PTQ2 截断：{what} 需要 [{off}, {off + n})，文件只有 {len(data)} B")


def read_ptq2_info(path) -> dict:
    data = Path(path).read_bytes()
    if len(data) < 20:
        raise ValueError(f"PTQ2 文件过短：{len(data)} B < 20 B")
    if data[:4] != MAGIC:
        raise ValueError(f"PTQ2 magic 不符：实得 {data[:4]!r}")
    n_conv, n_block, stem_pool, ncp = struct.unpack_from("<4i", data, 4)
    if n_conv <= 0:
        raise ValueError(f"PTQ2 n_conv 必须 > 0，实得 {n_conv}")
    if n_block < 0 or ncp < 0:
        raise ValueError(f"PTQ2 n_block/ncp 不能为负：{n_block}/{ncp}")
    if stem_pool not in (0, 1):
        raise ValueError(f"PTQ2 stem_pool 只允许 0/1，实得 {stem_pool}")

    off = 20
    convs = []
    for i in range(n_conv):
        _need(data, off, 40, f"conv[{i}].meta")
        meta = struct.unpack_from("<10i", data, off)
        off += 40
        cout, cin, kh, kw, sh, sw, ph, pw, relu, groups = meta
        if min(cout, cin, kh, kw) <= 0:
            raise ValueError(f"conv[{i}] cout/cin/kh/kw 必须为正：{meta[:4]}")
        if sh < 1 or sw < 1 or ph < 0 or pw < 0:
            raise ValueError(f"conv[{i}] stride/padding 非法：{meta[4:8]}")
        if relu not in (0, 1, 2):
            raise ValueError(f"conv[{i}] relu 只允许 0/1/2，实得 {relu}")
        if groups < 1:
            raise ValueError(f"conv[{i}] groups 必须 >= 1，实得 {groups}")
        nw = cout * cin * kh * kw
        _need(data, off, (nw + cout) * 4, f"conv[{i}].weight/bias")
        convs.append({
            "index": i, "cout": cout, "cin": cin, "kh": kh, "kw": kw,
            "sh": sh, "sw": sw, "ph": ph, "pw": pw,
            "relu": relu, "groups": groups,
            "weight_offset": off, "n_weights": nw,
            "bias_offset": off + nw * 4,
        })
        off += (nw + cout) * 4

    blocks = []
    for i in range(n_block):
        width = ncp + 3
        _need(data, off, width * 4, f"block[{i}]")
        vals = struct.unpack_from(f"<{width}i", data, off)
        off += width * 4
        idx = list(vals[:ncp])
        ds, stride, residual = vals[ncp:]
        # -1 补位只能集中在尾部
        seen_pad = False
        for k, ci in enumerate(idx):
            if ci == -1:
                seen_pad = True
                continue
            if seen_pad:
                raise ValueError(
                    f"block[{i}] 的 -1 补位不在尾部：位置 {k} 是 {ci}")
            if ci < 0 or ci >= n_conv:
                raise ValueError(f"block[{i}] conv 索引越界：{ci}")
        if ds < -1 or ds >= n_conv:
            raise ValueError(f"block[{i}] ds 越界：{ds}")
        if stride < 1:
            raise ValueError(f"block[{i}] stride < 1：{stride}")
        if residual not in (0, 1, 2):
            raise ValueError(f"block[{i}] residual 只允许 0/1/2：{residual}")
        blocks.append({"index": i, "convs": idx, "ds": ds,
                       "stride": stride, "residual": residual})

    _need(data, off, 8, "fc 头")
    fc_in, fc_out = struct.unpack_from("<2i", data, off)
    off += 8
    if fc_in <= 0 or fc_out <= 0:
        raise ValueError(f"fc_in/fc_out 必须为正：{fc_in}/{fc_out}")
    _need(data, off, (fc_in * fc_out + fc_out) * 4, "fc 权重/bias")
    fc_weight_offset = off
    off += (fc_in * fc_out + fc_out) * 4
    if off != len(data):
        raise ValueError(
            f"PTQ2 长度不符：解析终点 {off} B，文件 {len(data)} B"
            f"（尾部多余 {len(data) - off} B 或截断）")

    return {
        "path": str(path), "bytes": len(data),
        "n_conv": n_conv, "n_block": n_block, "ncp": ncp,
        "stem_pool": stem_pool,
        "convs": convs, "blocks": blocks,
        "fc_in": fc_in, "fc_out": fc_out,
        "fc_weight_offset": fc_weight_offset,
        "n_grouped": sum(1 for c in convs if c["groups"] != 1),
        "n_depthwise": sum(1 for c in convs
                           if c["cin"] == 1 and c["groups"] == c["cout"]),
        "n_relu1": sum(1 for c in convs if c["relu"] == 1),
        "n_relu6": sum(1 for c in convs if c["relu"] == 2),
        "n_relu0": sum(1 for c in convs if c["relu"] == 0),
        "n_residual1": sum(1 for b in blocks if b["residual"] == 1),
        "n_residual2": sum(1 for b in blocks if b["residual"] == 2),
        "n_residual0": sum(1 for b in blocks if b["residual"] == 0),
    }


def conv_weight(data: bytes, conv: dict):
    """从 read_ptq2_info 的 conv 记录里取出 [cout][cin][kh][kw] 的 float32 视图。"""
    import numpy as np
    arr = np.frombuffer(data, dtype="<f4", count=conv["n_weights"],
                        offset=conv["weight_offset"])
    return arr.reshape(conv["cout"], conv["cin"], conv["kh"], conv["kw"])
