# -*- coding: utf-8 -*-
"""四组数据的横向对比表（CIFAR-100+MobileNetV2 的 e6/e7 尚未跑完，留空）。"""
import glob
import io
import json
import os

RES = 'D:/WorkBuddy/ShowBro/lab/03-ptq-lab/results'
COMBOS = [
    ('cifar10_resnet18', 'ResNet-18'),
    ('cifar10_mobilenetv2', 'MobileNetV2'),
    ('cifar100_resnet18', 'ResNet-18'),
    ('cifar100_mobilenetv2', 'MobileNetV2'),
]


def latest(sub, exp):
    fs = sorted(glob.glob(os.path.join(RES, sub, '%s_*.json' % exp)))
    if not fs:
        return None
    try:
        return json.load(io.open(fs[-1], encoding='utf-8'))
    except Exception:
        return None


OUT = []
def w(s=''):
    OUT.append(s)

w('=' * 96)
w('四组（数据集 × 模型）横向对比  ·  2026-10-17')
w('=' * 96)
w()
w('%-18s %-8s %9s %10s %10s %11s %11s' %
  ('组合', 'FP32', 'W4A8', '掉点', 'AdaRound', '回收率', '混合精度'))
w('-' * 96)

for sub, arch in COMBOS:
    ds = sub.split('_')[0] + ' ' + sub.split('_')[1]
    e1 = latest(sub, 'e1_baseline')
    e2 = latest(sub, 'e2_rtn_ladder')
    e5 = latest(sub, 'e5_mixed')
    e6 = latest(sub, 'e6_adaround')
    e7 = latest(sub, 'e7_combo')

    fp = (e1 or {}).get('fp32_acc', 0) * 100
    lad = (e2 or {}).get('results') or {}
    w4a8 = lad.get('W4A8', 0) * 100
    drop = fp - w4a8

    ada = None
    if e6 and e6.get('iters') == 20000:
        ada = (e6['adaround_acc'] - e6['rtn_acc']) * 100
    rate = ('%.2f' % (ada / drop)) if (ada and drop) else '—'

    mix = None
    if e5:
        # e5 结构是 rows=[{k,acc,avg_bits,gain_vs_w4}]；挑 avg_bits<=5 里 acc 最高
        best = None
        for v in (e5.get('rows') or []):
            if not isinstance(v, dict):
                continue
            ab = v.get('avg_bits')
            if ab is None or ab > 5.0:
                continue
            if best is None or v.get('acc', 0) > best.get('acc', 0):
                best = v
        if best:
            mix = best.get('gain_vs_w4')
            if mix is None:
                mix = (best['acc'] * 100 - w4a8)
            else:
                mix = mix * 100

    w('%-18s %-8s %8.2f%% %9.2f%% %9.2fpp %10s %10s %10s'
      % (sub, arch, fp, w4a8, drop,
         ('%+.2fpp' % ada) if ada is not None else '—',
         rate,
         ('%+.2fpp' % mix) if mix is not None else '—'))

w('-' * 96)
w()
w('说明：')
w('  回收率 = AdaRound 收益 / W4A8 掉点')
w('  混合精度 = e5 中满足平均位宽<=5 的最优配置 vs 统一 W4A8')
w()
w('=' * 96)

txt = '\n'.join(OUT)
dst = 'D:/WorkBuddy/ShowBro/lab/03-ptq-lab/docs/_四组横向对比.txt'
io.open(dst, 'w', encoding='utf-8').write(txt)
print(txt)