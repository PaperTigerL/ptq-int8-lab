# -*- coding: utf-8 -*-
"""对比两个 PTQW 权重文件的结构，验证 09 导出的与 06 的二进制兼容。

golden test 的前置检查：格式一致 → 才能用 06 的引擎二进制跑对齐。

用法：
    python tests/check_binary_compat.py
"""
import struct
import sys
from pathlib import Path

def _find_lab() -> Path:
    """向上找到含 06-gpu-int8 的目录（即 lab/）。

    ⛔ 不要用 `parents[n]` 硬编码层数：Windows 路径（`D:\\...\\tests`）与
       WSL 挂载路径（`/mnt/d/.../tests`）的 parents 深度看起来一样，
       但一旦多一层或少一层目录就会指到错误位置 ——
       实测 `parents[1]` 指到了 09 自己，导致找不到文件却报
       OverflowError 而不是 FileNotFoundError，难查。
       按「目录里有什么」来找，与层级无关。
    """
    for p in Path(__file__).resolve().parents:
        if (p / "06-gpu-int8").is_dir() and (p / "03-ptq-lab").is_dir():
            return p
    raise FileNotFoundError("向上找不到 lab/（应含 06-gpu-int8 与 03-ptq-lab）")


LAB = _find_lab()
A = LAB / "06-gpu-int8" / "cpp" / "resnet18_folded.bin"
B = Path(__file__).resolve().parent.parent / "results" / "resnet18_folded_w8a8.bin"


def probe(path):
    """解析 PTQW，返回结构摘要。抛异常表示格式不对。"""
    d = path.read_bytes()
    assert d[:4] == b"PTQW", f"{path.name}: magic 不符"
    n_conv, n_block, pool = struct.unpack_from("<3i", d, 4)
    off = 4 + 12      # magic 4 字节 + 3 个 int32 各 4 字节 = 16
    # ⛔ 写成 `off = 12` 会漏掉 magic 的 4 字节，之后所有字段都错位一格，
    #    症状是 conv[0] 解出 (1,64,3,7,7,2,2,3,3) 这种明显不对的值，
    #    再往后就会 OverflowError。错误信息完全指不到真正的原因。
    convs, tot_w = [], 0
    for _ in range(n_conv):
        cout, cin, kh, kw, sh, sw, ph, pw, relu = struct.unpack_from("<9i", d, off)
        off += 36
        tot_w += cout * cin * kh * kw
        off += 4 * (cout * cin * kh * kw) + 4 * cout
        convs.append((cout, cin, kh, kw, sh, sw, ph, pw, relu))
    blocks = [struct.unpack_from("<4i", d, off + 16 * i) for i in range(n_block)]
    off += 16 * n_block
    fc_in, fc_out = struct.unpack_from("<2i", d, off)
    off += 8 + 4 * fc_in * fc_out + 4 * fc_out
    return dict(path=path, n_conv=n_conv, n_block=n_block, pool=pool,
                convs=convs, blocks=blocks, fc=(fc_in, fc_out),
                n_params=tot_w + fc_in * fc_out,
                consumed=off, size=len(d), ok=(off == len(d)))


def main():
    for p in (A, B):
        if not p.exists():
            print(f"[SKIP] {p} 不存在")
            return 0 if p is B else 2
    a, b = probe(A), probe(B)

    for tag, r in (("06(既有)", a), ("09(新导出)", b)):
        flag = "OK" if r["ok"] else "字节不匹配!"
        print(f"{tag:<14} conv={r['n_conv']} block={r['n_block']} pool={r['pool']} "
              f"params={r['n_params']:,} size={r['size'] / 1e6:.1f}MB  {flag}")

    # 逐项比对元信息（权重数值本身允许不同 —— 06 是它自己导出的）
    same = True
    for key in ("n_conv", "n_block", "pool", "fc", "n_params"):
        if a[key] != b[key]:
            print(f"  ✗ {key}: {a[key]} vs {b[key]}")
            same = False
    for i, (x, y) in enumerate(zip(a["convs"], b["convs"])):
        if x != y:
            print(f"  ✗ conv[{i}] 元信息不同: {x} vs {y}")
            same = False
            break
    for i, (x, y) in enumerate(zip(a["blocks"], b["blocks"])):
        if x != y:
            print(f"  ✗ block[{i}] 不同: {x} vs {y}")
            same = False
            break

    print()
    if same and b["ok"]:
        print("✓ 二进制格式一致 —— 09 导出的文件可直接喂给 06 的引擎二进制")
        return 0
    print("✗ 格式不一致，golden test 无法用同一份二进制对齐")
    return 1


if __name__ == "__main__":
    sys.exit(main())
