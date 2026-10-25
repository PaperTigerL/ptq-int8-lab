#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 ptq-lab dump 出的 scale 转成 ncnn 量化表。

这是「研究侧配置 → 部署侧配置」的桥。已从源码 + 实测双重证实：
ncnn2int8 对量化表是**照单执行**，不重算任何一个 scale
（ncnn2int8.cpp:246 直接取表值；mat.cpp:2420 把 scale 当 Quantize 层权重加载）。
实测：把 conv1 的权重 scale 乘 2 灌进去，精度从 87.05% 掉到 80.53%（-6.52pp）。

表格式（每层两行）：
    <ncnn层名>            <激活 scale>              # 单值，per-tensor
    <ncnn层名>_param_0    <权重 scale_0> ... <权重 scale_{C-1}>   # 逐输出通道

⚠️ 一个必须说明的结构性差异：ptq-lab 的激活量化是**非对称**的
   （a_scale + a_zero 两个参数），而 ncnn Convolution 的 int8 激活只有
   一个 scale，是**对称**的。本工具只用 a_scale、丢弃 a_zero。
   这不是疏漏，是引擎能力的边界，影响要在评测里量化，不能假装没发生。

⚠️⚠️ 最大的坑：两边的 scale 约定互为倒数。
   ptq-lab: scale = absmax / 127   （除数，w_q = round(w / scale)）
   ncnn   : scale = 127 / absmax   （乘数，ncnn2table.cpp:580）
   直接把 ptq 的 scale 原样写进 ncnn 表，模型会退化成随机猜。
   实测踩过：注入未取倒数的 scale，精度 87.05% -> 10.00%（十个类乱猜）。

用法：
    python scripts/make_table.py \\
        --scales ../../03-ptq-lab/results/cifar10_resnet18/scales_w8a8_minmax.json \\
        --param models/resnet18_cifar10_opt.param \\
        --baseline tables/resnet18_cifar10_kl.table \\
        --out tables/resnet18_cifar10_ptq_minmax.table
"""
import argparse
import json
import os
import re
import sys

# 码空间换算系数：ptq 非对称 255 码 -> ncnn 对称 ±127 码。
# 由来与实测见 docs/阶段2-PTQ配置迁移到ncnn.md §3.6。
# 用 --no-sym-fix 可以关掉（复现修正前的旧行为）。
SYM_CODE_FIX = 127.0 / 255.0


def log(m):
    print(m, flush=True)


def parse_param_layers(param_path):
    """从 ncnn param 文件里取出所有需要量化的层名（Convolution / InnerProduct）。

    以 param 为准而不是硬编码层名：param 是引擎侧的唯一事实来源，
    模型换了层名也不会错。
    """
    layers = []
    with open(param_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            ltype, name = parts[0], parts[1]
            if ltype in ("Convolution", "ConvolutionDepthWise", "InnerProduct"):
                layers.append((ltype, name))
    if not layers:
        raise RuntimeError("%s 里没找到 Convolution/InnerProduct 层" % param_path)
    return layers


def ptq_to_ncnn(name):
    """PyTorch 模块路径 → ncnn 层名。

    conv1                    -> /conv1/Conv
    layer1.0.conv1           -> /layer1/layer1.0/conv1/Conv
    layer2.0.downsample.0    -> /layer2/layer2.0/downsample/downsample.0/Conv
    fc                       -> /fc/Gemm

    两个容易错的地方：
    1. layer1.0 这个「stage.block」在 ncnn 里是**两级目录**（/layer1/layer1.0/），
       所以不能简单地 replace('.', '/')。
    2. 多段模块路径（downsample.0）会被 onnx2ncnn 展开成「目录/全名」两级：
       downsample.0 -> downsample/downsample.0。单段路径（conv1）不展开。
    """
    if name == "fc":
        return "/fc/Gemm"
    parts = name.split(".")
    if parts[0] == "conv1":
        return "/conv1/Conv"
    if parts[0].startswith("layer") and len(parts) >= 3:
        stage = parts[0]                       # layer1
        block = "%s.%s" % (parts[0], parts[1])  # layer1.0
        rest = parts[2:]                        # conv1 / downsample.0
        if len(rest) == 1:
            rest_path = rest[0]
        else:
            # downsample.0 -> downsample/downsample.0
            rest_path = "/".join(rest[:-1]) + "/" + ".".join(rest)
        return "/%s/%s/%s/Conv" % (stage, block, rest_path)
    raise ValueError(
        "无法把 PyTorch 层名 %r 映射到 ncnn 层名。已知模式只有 "
        "conv1 / layerN.M.xxx / fc 三种，新的命名要显式加规则 —— "
        "静默映射错比报错危险得多。" % name)


def read_baseline_table(path):
    """读 KL 基线表，返回 {key: [float, ...]}。用于回退 ptq 没覆盖的层。"""
    table = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            table[parts[0]] = [float(v) for v in parts[1:]]
    return table


def fmt(vals):
    return " ".join("%.6f" % v for v in vals)


def main():
    ap = argparse.ArgumentParser(
        description="把 ptq-lab 的 scale 转成 ncnn 量化表（照单执行，ncnn 不重算）")
    ap.add_argument("--scales", required=True, help="dump_scales.py 产出的 json")
    ap.add_argument("--param", required=True, help="ncnn .param 文件（层名的唯一事实来源）")
    ap.add_argument("--baseline", required=True,
                    help="KL 基线表，用于回退 ptq 未覆盖的层（如 fc）")
    ap.add_argument("--out", required=True, help="输出 .table 路径")
    ap.add_argument("--keep-fp16", default="",
                    help="要保护成 FP16 的层，PyTorch 层名，逗号分隔。"
                         "这些层不出现在表里 -> ncnn2int8 的 "
                         "quantize_convolution() 第 169-171 行 continue -> "
                         "该层保持 fp16 权重、激活不量化。"
                         "典型用法：--keep-fp16 layer3.0.conv2,layer2.0.downsample.0")
    ap.add_argument("--no-sym-fix", action="store_true",
                    help="关掉「asym 255 码 -> ncnn ±127 码」的换算（默认开）。"
                         "只用于复现 2026-10-18 修正前的旧行为，正常别用。")
    ap.add_argument("--a-symmetric", action="store_true",
                    help="激活是对称量化（json 的 meta.a_symmetric 已写明时会自动开）。"
                         "对称时注入 1/a_scale，不做 127/255 换算 —— "
                         "那个换算只适用于 asym 的 2^b 码铺满整段量程。")
    ap.add_argument("--a-asymmetric", action="store_true",
                    help="激活是非对称量化：除 scale 外再写一行 <层名>_zp（zero_point），"
                         "供 ncnn2int8 把 zp*sum(q_w) 折叠进 bias。"
                         "不给这个开关就不会多写行，阶段 2 的对​​称行为不受影响。")
    args = ap.parse_args()

    global SYM_CODE_FIX
    if args.no_sym_fix:
        SYM_CODE_FIX = 1.0
    log("码空间换算: %s（注入的激活 scale × %.6f）"
        % ("关闭（旧行为）" if args.no_sym_fix else "127/255（asym -> sym）",
           SYM_CODE_FIX))

    with open(args.scales) as f:
        payload = json.load(f)
    ptq_layers = payload["layers"]

    # 激活码空间：json 里写明就跟着走，命令行可显式覆盖
    a_symmetric = bool(payload.get("meta", {}).get("a_symmetric", False)) or args.a_symmetric
    log("激活码空间: %s" % ("对称（注入 1/a_scale）" if a_symmetric
                          else "非对称 -> 对称（注入 127/255 × 1/a_scale）"))
    meta = payload["meta"]

    ncnn_layers = parse_param_layers(args.param)
    baseline = read_baseline_table(args.baseline)

    log("ptq 配置 : %s" % args.scales)
    log("  口径   : W%dA%d，激活定标 %s，BN 已折 %s"
        % (meta["w_bits"], meta["a_bits"], meta["act_strategy"], meta["bn_folded"]))
    log("ncnn 层  : %d 个（来自 %s）" % (len(ncnn_layers), args.param))
    log("KL 基线  : %s（%d 行）" % (args.baseline, len(baseline)))

    # ---- 1. 层名映射，全部断言通过才算数 ----
    log("")
    log("[1/3] 层名映射（PyTorch -> ncnn）")
    ncnn_names = set(n for _, n in ncnn_layers)
    mapped = {}
    for ptq_name in sorted(ptq_layers):
        ncnn_name = ptq_to_ncnn(ptq_name)
        if ncnn_name not in ncnn_names:
            raise RuntimeError(
                "映射结果 %r（来自 PyTorch 层 %r）不在 param 的层名表里。\n"
                "  param 里的层：%s\n"
                "这说明某一侧的层名理解有误，此时产出的表不可信，已中止。"
                % (ncnn_name, ptq_name, sorted(ncnn_names)))
        mapped[ncnn_name] = ptq_name
    log("  %d 层全部映射成功" % len(mapped))

    # ---- 1.5 保护层：这些层不进表，ncnn2int8 会让它们保持 FP16 ----
    keep_fp16 = set()
    for nm in [s.strip() for s in args.keep_fp16.split(",") if s.strip()]:
        ncnn_name = ptq_to_ncnn(nm)
        if ncnn_name not in ncnn_names:
            raise RuntimeError(
                "--keep-fp16 里的 %r 映射到 %r，不在 param 的层名表里"
                % (nm, ncnn_name))
        if ncnn_name not in mapped:
            raise RuntimeError(
                "--keep-fp16 里的 %r（-> %r）本来就不在 ptq 配置里，无需保护"
                % (nm, ncnn_name))
        keep_fp16.add(ncnn_name)
    if keep_fp16:
        log("")
        log("[1.5] 保护成 FP16 的层（%d 个）：" % len(keep_fp16))
        for n in sorted(keep_fp16):
            log("    %s" % n)

    # ---- 2. 组装表 ----
    log("[2/3] 组装表")
    out_lines = []
    injected, fallback, protected = 0, 0, 0
    zp_written = 0

    for ltype, ncnn_name in ncnn_layers:
        w_key = "%s_param_0" % ncnn_name
        a_key = ncnn_name

        # 保护层：整条不进表。ncnn2int8 的 quantize_convolution() 靠
        # blob_int8scale_table.find(name) 找层，找不到就 continue，
        # 该层于是保持 fp16 权重 + 激活不量化。
        if ncnn_name in keep_fp16:
            protected += 1
            continue

        if ncnn_name in mapped:
            entry = ptq_layers[mapped[ncnn_name]]

            # 权重 scale 个数必须等于输出通道数，否则广播会错
            if len(entry["w_scale"]) != entry["out_channels"]:
                raise RuntimeError("%s 的 w_scale 有 %d 个，但 out_channels=%d"
                                   % (ncnn_name, len(entry["w_scale"]),
                                      entry["out_channels"]))

            if "a_scale" not in entry:
                raise RuntimeError(
                    "%s 缺 a_scale。dump_scales.py 应当为每个 Conv2d 产出激活 scale，"
                    "缺了说明校准没覆盖到，不能拿权重 scale 单独出一张表。"
                    % ncnn_name)

            # 非对称 -> 对称：只用 a_scale，a_zero 在此丢弃（引擎能力边界）
            # 除数 -> 乘数：ncnn 存 127/absmax，ptq 存 absmax/127，互为倒数
            #
            # 码空间换算（2026-10-18 修掉的一个真 bug）：
            #   ptq 侧是非对称，qmax = 2^bits - 1 = 255，scale = (hi-lo)/255，
            #   即把 [lo,hi] 铺满 [0,255] 这 256 个码字，靠 zero_point 平移。
            #   ncnn 侧是对称 int8，Quantize 层 float2int8() 截断在 ±127
            #   （src/layer/quantize.cpp:30-36），且**没有 zero_point**，
            #   表示区间被钉死在以 0 为中心的 [-127/s, +127/s]。
            #   直接注入 1/a_scale = 255/(hi-lo)，ncnn 就只能表示到 (hi-lo)/2，
            #   超出的一律截成 127 —— 桥自带 2 倍溢出。
            #   正确做法：让策略自己选的阈值恰好落在 ±127 上，
            #   即注入 127/(hi-lo) = (127/255) × (1/a_scale)。
            #
            # 对称量化（--a-symmetric / meta.a_symmetric）走另一条：
            #   ptq 的 a_scale = t / (2^(b-1)-1)，已经是「对称网格的步长」，
            #   ncnn 要的乘数正是它的倒数，不需要任何码空间换算。
            #   套用 127/255 会让 ncnn 只能表示量程的一半 —— 差 2 倍。
            if a_symmetric or args.a_asymmetric:
                # 码空间一致，直接取倒数：
                #   对称  : ptq a_scale = t/(2^(b-1)-1)，ncnn 乘数 = 1/a_scale
                #   非对称: ptq a_scale = (hi-lo)/(2^b-1)，ncnn 乘数 = 1/a_scale
                # 只有「ptq 的 asym 塞进 ncnn 的对称码空间」才要 127/255 换算。
                a_scale = [1.0 / entry["a_scale"]]
            else:
                a_scale = [SYM_CODE_FIX / entry["a_scale"]]
            w_scale = [1.0 / v for v in entry["w_scale"]]
            injected += 1

            # 非对称：多写一行 <层名>_zp，指向 ncnn2int8 做 bias 折叠。
            # 注意 key 是 a_key（ncnn 层名），不是 a_key + "_param_0"。
            if args.a_asymmetric and "a_zero" in entry:
                out_lines.append("%s_zp %.0f" % (a_key, entry["a_zero"]))
                zp_written += 1
        else:
            # ptq 没覆盖的层（如 fc/Gemm）：回退到 KL 基线，并明确告知
            if w_key not in baseline or a_key not in baseline:
                raise RuntimeError(
                    "%s 既不在 ptq 配置里，也不在 KL 基线表里，无从取值。"
                    "要么补 dump，要么补基线表。" % ncnn_name)
            w_scale = baseline[w_key]
            a_scale = baseline[a_key]
            fallback += 1
            log("  [回退] %-46s 用 KL 基线值（ptq 未覆盖此层）" % ncnn_name)

        out_lines.append("%s %s" % (a_key, fmt(a_scale)))
        out_lines.append("%s %s" % (w_key, fmt(w_scale)))

    # ---- 3. 写出 + 自检 ----
    log("[3/3] 写出并自检")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    # 自检：行数 = 2 × 进表的层数；每个 key 只出现一次
    n_in_table = len(ncnn_layers) - protected
    n_expect = 2 * n_in_table + zp_written
    if len(out_lines) != n_expect:
        raise RuntimeError("产出了 %d 行，期望 %d 行（%d 层×2 + %d 行 zp）"
                           % (len(out_lines), n_expect, n_in_table, zp_written))
    seen = {}
    for line in out_lines:
        k = line.split()[0]
        seen[k] = seen.get(k, 0) + 1
    dupes = [k for k, c in seen.items() if c != 1]
    if dupes:
        raise RuntimeError("有 key 重复出现：%s" % dupes)

    # 保护层必须确实不在表里 —— 这是 --keep-fp16 的语义底线
    for n in keep_fp16:
        if n in seen or ("%s_param_0" % n) in seen:
            raise RuntimeError("保护层 %s 竟然出现在表里，--keep-fp16 失效" % n)

    log("")
    log("写出 %s" % args.out)
    log("  %d 行 = %d 层 × 2（激活 + 权重）" % (len(out_lines), n_in_table))
    log("  注入 ptq scale : %d 层" % injected)
    log("  回退 KL scale  : %d 层" % fallback)
    log("  保护成 FP16    : %d 层" % protected)
    if fallback:
        log("")
        log("  ⚠️ 上面 %d 层用的是 KL 值而非 ptq 值，报数字时不能算作"
            "「ptq 配置完整迁移」。" % fallback)
    if protected:
        log("")
        log("  ⚠️ %d 层保持 FP16。它们不参与 INT8，"
            "报「混合精度」时要说清是 INT8+FP16 混合，不是 W4A8。" % protected)
    log("")
    log("下一步：ncnn2int8 <param> <bin> <out_param> <out_bin> %s" % args.out)


if __name__ == "__main__":
    main()
