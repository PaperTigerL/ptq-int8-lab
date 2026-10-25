#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨模型对比：ResNet-18 vs MobileNetV2（均 CIFAR-10，各用自己的 50ep 基线）。

回答一个问题：**现有结论哪些是模型无关的通则，哪些只是 ResNet-18 的特例？**
"""
import csv
import io
import os

RES = 'D:/WorkBuddy/ShowBro/lab/03-ptq-lab/results'
OUT = []


def w(s=""):
    OUT.append(s)


def load(sub):
    p = os.path.join(RES, sub, 'all.csv')
    if not os.path.isfile(p):
        return []
    return list(csv.DictReader(io.open(p, encoding='utf-8-sig')))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


R18 = load('cifar10_resnet18')
RMB = load('cifar10_mobilenetv2')

w('=' * 78)
w('跨模型验证 · ResNet-18 vs MobileNetV2（均 CIFAR-10）')
w('=' * 78)
w()

fp18 = max([fnum(r['fp32_acc']) for r in R18 if r.get('exp') == 'e1_baseline'] or [0])
fpmb = max([fnum(r['fp32_acc']) for r in RMB if r.get('exp') == 'e1_baseline'] or [0])

w('【0】基线')
w('-' * 78)
w('  ResNet-18   : %.2f%%   11.18M params   20 Conv2d（0 depthwise）' % (fp18 * 100))
w('  MobileNetV2 : %.2f%%    2.24M params   52 Conv2d（17 depthwise）' % (fpmb * 100))
w('  两者 BN 折叠均数学等价（差异 0.000000% / 2.7e-16）')
w()


# ---------------- e2 位宽阶梯 ----------------
def ladder(rows):
    """e2 拍平后的列名形如  results.W8A8=0.8388。"""
    d = {}
    for r in rows:
        if r.get('exp') != 'e2_rtn_ladder':
            continue
        for k, v in r.items():
            if k.startswith('results.'):
                cfg = k[len('results.'):]
                val = fnum(v)
                if cfg and val is not None:
                    d[cfg] = val
    return d


L18 = ladder(R18)
LMB = ladder(RMB)

w('【1】RTN 位宽阶梯 —— 结论「位宽每降 1 档掉多少」跨模型是否成立')
w('-' * 78)
if L18 and LMB:
    w('  %-8s %>10s %12s' % ('配置', 'ResNet-18', 'MobileNetV2') if False else
      '  %-8s %11s %13s' % ('配置', 'ResNet-18', 'MobileNetV2'))
    w('  ' + '-' * 36)
    for cfg in ['W8A8', 'W8A32', 'W4A8', 'W4A32', 'W4A4', 'W3A8', 'W3A4']:
        a, b = L18.get(cfg), LMB.get(cfg)
        w('  %-8s %11s %13s' % (
            cfg,
            ('%.2f%%' % (a * 100)) if a is not None else '  -',
            ('%.2f%%' % (b * 100)) if b is not None else '  -'))
else:
    w('  （e2 数据结构不匹配，实际读到的配置：%s / %s）' % (list(L18)[:4], list(LMB)[:4]))
w()


# ---------------- 掉点归因 ----------------
def drop(L, hi, lo):
    if hi in L and lo in L:
        return (L[lo] - L[hi]) * 100
    return None


if L18 and LMB:
    w('【2】掉点归因 —— 结论 A「激活 8bit 几乎免费」是否跨模型成立')
    w('-' * 78)
    w('  %-32s %11s %13s' % ('变化', 'ResNet-18', 'MobileNetV2'))
    w('  ' + '-' * 58)
    pairs = [
        ('权重 8->4（激活不动）', 'W8A32', 'W4A32'),
        ('激活 FP32->A8（权重 W4）', 'W4A32', 'W4A8'),
        ('激活 A8->A4（权重 W4）', 'W4A8', 'W4A4'),
        ('权重 4->3（激活 A8）', 'W4A8', 'W3A8'),
    ]
    for name, hi, lo in pairs:
        a, b = drop(L18, hi, lo), drop(LMB, hi, lo)
        w('  %-32s %11s %13s' % (
            name,
            ('%+.2f pp' % a) if a is not None else '-',
            ('%+.2f pp' % b) if b is not None else '-'))
    w()


# ---------------- e4 敏感度 ----------------
def sens(sub, exp='e4_sensitivity'):
    """**必须从 json 读** —— csv 里的 sens 是 list，被整串拍平成字符串，
    没法解析出逐层的 name/delta（实测踩过）。json 才是唯一事实来源。

    ⚠️ 字段用 `delta_pct`（掉点百分点，正值 = 敏感），不是 `delta`（原始比例）。
       且排序是**降序**（掉得最多的排前面）。取错字段/方向会让结论完全反过来。
    """
    import glob
    import json as _json
    out = []
    for f in sorted(glob.glob(os.path.join(RES, sub, '%s_*.json' % exp))):
        with io.open(f, encoding='utf-8') as fh:
            d = _json.load(fh)
        for item in (d.get('sens') or []):
            if isinstance(item, dict) and 'name' in item:
                v = item.get('delta_pct')
                if v is None:
                    v = (item.get('delta') or 0.0) * 100
                out.append((item['name'], float(v)))
    return sorted(out, key=lambda x: -x[1])


S18 = sens('cifar10_resnet18')
SMB = sens('cifar10_mobilenetv2')

w('【3】逐层敏感度 —— 结论 C「首层是唯一硬骨头」是否跨模型成立')
w('-' * 78)
if S18 or SMB:
    w('  %-30s | %-30s' % ('ResNet-18（20 层）', 'MobileNetV2（52 层）'))
    w('  ' + '-' * 66)
    for i in range(5):
        a = ('%.2f%%  %s' % (S18[i][1], S18[i][0][:20])) if i < len(S18) else ''
        b = ('%.2f%%  %s' % (SMB[i][1], SMB[i][0][:20])) if i < len(SMB) else ''
        w('  %-30s | %-30s' % (a, b))
    w()
    if S18 and SMB:
        w('  敏感度极差（e4 实测 spread_pct）：')
        w('    ResNet-18   : %.2f pp  （最敏感 %s %.2f%%）'
          % (S18[0][1] - S18[-1][1], S18[0][0], S18[0][1]))
        w('    MobileNetV2 : %.2f pp  （最敏感 %s %.2f%%）'
          % (SMB[0][1] - SMB[-1][1], SMB[0][0], SMB[0][1]))
        w()
        w('  ⭐ 首层是否仍是最敏感层？')
        for tag, lst, first in [('ResNet-18  ', S18, 'conv1'),
                                ('MobileNetV2', SMB, 'features.0.0')]:
            if lst:
                top = lst[0][0]
                w('    %s 最敏感层 = %-22s %.2f%%   %s'
                  % (tag, top, lst[0][1],
                     '✓ 就是首层' if top == first else '✗ 不是首层'))
        w()
        # 首层 vs 次敏感层的差距 —— 判断「唯一硬骨头」是否成立
        w('  「首层独大」是否成立（第一名的领先幅度）：')
        for tag, lst in [('ResNet-18  ', S18), ('MobileNetV2', SMB)]:
            if len(lst) >= 2:
                gap = lst[0][1] - lst[1][1]
                ratio = (lst[0][1] / lst[1][1]) if lst[1][1] else float('inf')
                w('    %s 第1名 %.2f%% vs 第2名 %.2f%%  领先 %.2f pp（%.1f 倍）'
                  % (tag, lst[0][1], lst[1][1], gap, ratio))
        w()
        # depthwise 层整体表现
        dw = [(n, d) for n, d in SMB if '.conv.0.0' in n or n == 'features.0.0']
        if dw:
            dw_sorted = sorted(dw, key=lambda x: -x[1])
            w('  MobileNetV2 的 depthwise 层（conv.0.0）掉点：')
            for n, d in dw_sorted[:5]:
                w('    %-26s %.2f%%' % (n[:26], d))
else:
    w('  （未读到敏感度数据）')
w()
w('=' * 78)

txt = '\n'.join(OUT)
dst = 'D:/WorkBuddy/ShowBro/lab/03-ptq-lab/docs/跨模型验证结果.md'
io.open(dst, 'w', encoding='utf-8').write(txt)
print(txt)
print('\n已写入: %s' % dst)
