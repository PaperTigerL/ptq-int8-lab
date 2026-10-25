# -*- coding: utf-8 -*-
"""检查点链审计：识别「说续跑、实际从零重训」以及跨协议混编。

判定依据不是日志文字，而是每个 ckpt 里的元数据 + 相邻 epoch 的 fc 权重
余弦相似度：连续训练时 fc 变化平缓（通常 > 0.95），重新初始化会掉到 0 附近
（ImageNet 起点固定，conv1 余弦仍接近 1 —— 所以只看 conv1 会漏判）。

用法：
    /home/user/pyenv-torch/bin/python tools/audit_ckpts.py \
        --dir /mnt/d/WorkBuddy/ShowBro/paper-code/ckpts [--arch resnet50]

退出码：0 = 至少有一个可引用链；1 = 没有 ckpt；不因 RESTART 单独失败。
"""
import argparse
import re
from pathlib import Path

import torch

FC_COS_BREAK = 0.5


def cos(a, b):
    return float((a.flatten() * b.flatten()).sum()
                 / (a.norm() * b.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--arch", default="resnet50")
    args = ap.parse_args()

    d = Path(args.dir)
    pat = re.compile(rf"ckpt_{re.escape(args.arch)}(?:_([A-Za-z0-9._-]+))?_ep(\d+)\.pth$")
    rows = []
    for p in sorted(d.glob(f"ckpt_{args.arch}*.pth")):
        m = pat.match(p.name)
        if not m:
            continue
        c = torch.load(p, map_location="cpu", weights_only=False)
        rows.append(dict(tag=m.group(1) or "", ep=int(c["epoch"]), path=p,
                         test=float(c.get("test_acc", float("nan"))),
                         protocol=c.get("protocol"),
                         target=c.get("target_epoch"),
                         resumed_from=c.get("resumed_from"),
                         resume_diff=c.get("resume_max_abs_diff"),
                         init_from=c.get("init_from"),
                         sd=c["model"]))
    if not rows:
        print(f"[审计] {d} 下没有 ckpt_{args.arch}*.pth")
        return 1

    rows.sort(key=lambda r: (r["tag"], r["ep"]))
    print(f"[审计] {len(rows)} 个检查点 · {d}")
    print(f"{'tag':<10} {'ep':>3} {'test%':>7} {'target':>6} "
          f"{'resume':>7} {'init_ep':>7} {'fc_cos':>8}  chain/protocol")
    prev = None
    for r in rows:
        c = cos(prev["sd"]["fc.weight"], r["sd"]["fc.weight"]) \
            if prev and prev["tag"] == r["tag"] else float("nan")
        flag = ""
        if c == c and c < FC_COS_BREAK:
            flag = "  <-- RESTART"
        print(f"{r['tag'] or '-':<10} {r['ep']:>3} {r['test']:>7.2f} "
              f"{str(r['target']):>6} {str(r['resumed_from'] or '-'):>7} "
              f"{(('ep' + str(r.get('init_epoch'))) if r.get('init_from') else '-'):>7} "
              f"{c:>8.4f}  {r['protocol']}{flag}")
        prev = r

    print("-" * 78)
    # 每个 tag 内按「fc 余弦连续 + 协议相同」切段，逐段列出，避免只看最后一段
    all_best = {}
    for tag in sorted({r["tag"] for r in rows}):
        chain = [r for r in rows if r["tag"] == tag]
        segs, cur = [], [chain[0]]
        for prev_r, r in zip(chain, chain[1:]):
            c = cos(prev_r["sd"]["fc.weight"], r["sd"]["fc.weight"])
            if (r["ep"] != prev_r["ep"] + 1
                    or c < FC_COS_BREAK
                    or prev_r.get("protocol") != r.get("protocol")):
                segs.append(cur)
                cur = [r]
            else:
                cur.append(r)
        segs.append(cur)
        for seg in segs:
            best = max(seg, key=lambda r: r["test"])
            print(f"[链] tag={tag or '-'} ep{seg[0]['ep']}..ep{seg[-1]['ep']}"
                  f"（{len(seg)} 个） protocol={seg[-1]['protocol']}")
            print(f"     段内 best: ep{best['ep']} test={best['test']:.2f}% "
                  f"({best['path'].name})；最后: ep{seg[-1]['ep']} "
                  f"test={seg[-1]['test']:.2f}%")
            if seg[-1].get("resume_diff") not in (None, 0.0):
                print(f"     ⚠️ resume_max_abs_diff={seg[-1]['resume_diff']} != 0")
            if seg[-1].get("resumed_from"):
                print(f"     续跑来源: {seg[-1]['resumed_from']}")
            if seg[-1].get("init_from"):
                print(f"     权重初始化来源: {seg[-1]['init_from']} "
                      f"ep{seg[-1].get('init_epoch')} "
                      f"test={seg[-1].get('init_test_acc')}%")
            pr = seg[-1]["protocol"]
            if pr not in all_best or best["test"] > all_best[pr]["test"]:
                all_best[pr] = best
    print("-" * 78)
    for pr, b in all_best.items():
        print(f"[可引用 best] {pr}: ep{b['ep']} test={b['test']:.2f}% "
              f"({b['path'].name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
