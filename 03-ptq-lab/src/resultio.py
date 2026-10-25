# -*- coding: utf-8 -*-
"""结果落盘的公共部分。

**为什么单独拆一个文件**：all.csv 的重建脚本（tools/rebuild_csv.py）应该在任何环境下
都能跑（比如没装 torch 的机器上只想看看历史数据），不能因为 import 不到 torch 就崩。
所以这个文件只依赖标准库。
"""
import csv
import json
import os


def flatten_payload(payload, prefix=""):
    """把任意嵌套的结果字典拍平成 {列名: 标量}。

    - 嵌套 dict 用点号连接（results.2. W4A8 RTN.acc）
    - list / 其他类型序列化成 JSON 字符串

    坑（老版本的写法）：只拍平一层，于是 e7 的 results（dict 套 dict）、
    ladder / robust（list）会被整段静默丢掉 —— 文件看起来在落盘，其实数据没了。
    """
    out = {}
    for k, v in payload.items():
        key = "%s.%s" % (prefix, k) if prefix else k
        if isinstance(v, (int, float, str, bool)):
            out[key] = v
        elif isinstance(v, dict):
            out.update(flatten_payload(v, key))
        elif isinstance(v, list):
            out[key] = json.dumps(v, ensure_ascii=False)
        elif v is None:
            out[key] = ""
        else:
            out[key] = str(v)
    return out


def append_csv_row(csv_path, row):
    """追加一行，并保持全表列对齐。

    坑（老版本的写法）：每次用「当前行的键」当 fieldnames 直接 append，
    于是不同实验的列互相错位 —— e2 的 1024（校准集大小）被塞进了 fp32_acc 那一列。
    正确做法：读回全部旧行 → 取列名并集 → 整表重写。
    """
    rows = []
    keys = []
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, encoding="utf-8", newline="") as f:
            rdr = csv.DictReader(f)
            keys = list(rdr.fieldnames or [])
            for r in rdr:
                rows.append(dict(r))
    for k in row:
        if k not in keys:
            keys.append(k)
    rows.append(row)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=keys, restval="",
                             extrasaction="ignore")
        wtr.writeheader()
        for r in rows:
            wtr.writerow(r)
    return keys
