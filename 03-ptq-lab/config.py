# -*- coding: utf-8 -*-
"""PTQ 实验平台 · 集中配置

所有可调参数都放这里，实验脚本不再各自硬编码。
改这一个文件就能换模型、换数据集、换校准集大小。

================================================================================
2026-10-15 参数化：支持换数据集 / 换模型
================================================================================
新增两个开关，其余量全部由它们推导：

    DATASET = "cifar10" | "cifar100"
    ARCH    = "resnet18" | "mobilenetv2"

用法：
    import config as C
    C.configure("cifar100", "resnet18")     # 在跑之前调用一次
    # 之后 C.CKPT_MAIN / C.NUM_CLASSES / C.MEAN / C.STD 等全部自动跟着变

    # 命令行也可以（./run.sh 会转成环境变量）：
    PTQ_DATASET=cifar100 PTQ_ARCH=mobilenetv2 ./run.sh all

**重要口径**：本平台的 ResNet-18 是 **ImageNet 版 stem**（7×7 stride 2 + maxpool，
11.18M 参数），不是 CIFAR 版的 3×3 stem。已实测确认（见下方 RESNET_STEM_NOTE）。
换数据集时**必须保持同一版本**，否则两组结果不可比。
"""
import os
import torch

BASE = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# 数据来源：复用 paper-code 已有的资产
# ==============================================================================
PAPER_CODE = "/mnt/d/WorkBuddy/ShowBro/paper-code"
DATA_DIR = os.path.join(PAPER_CODE, "data")

RESNET_STEM_NOTE = (
    "本平台的 ResNet-18 用的是 torchvision 默认 stem：conv1 = Conv2d(3,64,7×7,stride=2,padding=3) "
    "+ maxpool，11.18M 参数。**不是** CIFAR 常用的 3×3 stride1 stem。"
    "这是有意保持的：换了 stem 就换了模型，跨数据集对比会失去意义。"
)

# ==============================================================================
# 数据集与模型定义（单一事实来源）
# ==============================================================================
DATASET_SPECS = {
    "cifar10": {
        "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "cifar100": {
        "num_classes": 100,
        "mean": (0.5071, 0.4865, 0.4409),
        "std": (0.2673, 0.2564, 0.2762),
    },
}

ARCH_SPECS = {
    "resnet18": {"params_m": 11.18, "kind": "cifar-stem"},
    "mobilenetv2": {"params_m": 2.24, "kind": "imagenet-stem"},
}

# 权重命名约定。
#⚠️ 默认组合（cifar10 + resnet18）沿用**历史路径** paper-code/ 下那两个文件——
#   它们是 87.16% 的来源，不能挪，一挪整个平台就跑不起来了。
#   其他新组合统一放 paper-code/ckpts/。
_CKPT_LEGACY = {
    ("cifar10", "resnet18"): {
        "main": "cifar_resnet18_50ep.pth",
        "weak": "cifar_resnet18.pth",
    },
}
CKPT_DIR = os.path.join(PAPER_CODE, "ckpts")
CKPT_MAIN_NAME = "{ds}_{arch}_50ep.pth"
CKPT_WEAK_NAME = "{ds}_{arch}_3ep.pth"      # 欠拟合对照，仅 e1 用


def configure(dataset=None, arch=None):
    """按数据集/模型重新推导全部派生量。调用它之后 C.* 里的相关字段才可信。

    不传参则沿用环境变量 PTQ_DATASET / PTQ_ARCH（默认 cifar10 + resnet18）。
    """
    dataset = dataset or os.environ.get("PTQ_DATASET", "cifar10")
    arch = arch or os.environ.get("PTQ_ARCH", "resnet18")
    if dataset not in DATASET_SPECS:
        raise ValueError("未知数据集 %r，可选：%s" % (dataset, list(DATASET_SPECS)))
    if arch not in ARCH_SPECS:
        raise ValueError("未知模型 %r，可选：%s" % (arch, list(ARCH_SPECS)))

    spec = DATASET_SPECS[dataset]
    legacy = _CKPT_LEGACY.get((dataset, arch))
    if legacy:
        # 历史路径：直接指向 paper-code/ 根目录
        ckpt_main = os.path.join(PAPER_CODE, legacy["main"])
        ckpt_weak = os.path.join(PAPER_CODE, legacy["weak"])
    else:
        ckpt_main = os.path.join(CKPT_DIR, CKPT_MAIN_NAME.format(ds=dataset, arch=arch))
        ckpt_weak = os.path.join(CKPT_DIR, CKPT_WEAK_NAME.format(ds=dataset, arch=arch))

    globals()["DATASET"] = dataset
    globals()["ARCH"] = arch
    globals()["NUM_CLASSES"] = spec["num_classes"]
    globals()["MEAN"] = spec["mean"]
    globals()["STD"] = spec["std"]
    globals()["CIFAR_MEAN"] = spec["mean"]
    globals()["CIFAR_STD"] = spec["std"]
    globals()["CKPT_MAIN"] = ckpt_main
    globals()["CKPT_WEAK"] = ckpt_weak
    globals()["RESULT_DIR"] = os.path.join(
        BASE, "results", "%s_%s" % (dataset, arch))
    globals()["FIG_DIR"] = os.path.join(RESULT_DIR, "figs")
    globals()["CSV_PATH"] = os.path.join(RESULT_DIR, "all.csv")
    globals()["TAG"] = "%s_%s" % (dataset, arch)
    return dataset, arch


# 默认组合立刻生效（后续所有 import C.* 的代码都拿到已推导好的值）。
# ⚠️ **RESULT_DIR / FIG_DIR / CSV_PATH / TAG 由 configure() 写进 globals()**，
#    这里**绝不能再赋值** —— 否则会把 configure 刚推导好的带组合路径覆盖回
#    results/ 根目录，导致换模型的实验和新数据混在一起（2026-10-17 实测踩过）。
configure(os.environ.get("PTQ_DATASET", "cifar10"),
          os.environ.get("PTQ_ARCH", "resnet18"))

# ==============================================================================
# 数据与校准
# ==============================================================================
CALIB_SIZE = 1024     # 校准集图片数（激活量化范围靠它定）
CALIB_SEED = 2026     # 固定种子，保证实验可复现
CALIB_BATCH = 64
EVAL_BATCH = 256
NUM_WORKERS = 4

# ==============================================================================
# 量化参数（算子与实验共用，改这里就全局生效）
# ==============================================================================
DEFAULT_W_BITS = 8          # 权重量化默认位宽
DEFAULT_A_BITS = 8          # 激活量化默认位宽
PERCENTILE = 0.999          # percentile 校准：掐掉两端各 (1-p)，工程上最常用的折中

# scale_mse / scale_entropy 的搜索设置
#   num_candidates  候选上界个数；越大搜得越细，代价是线性变慢
#   shrink          候选上界从 absmax 等比缩到 absmax * shrink
#   num_bins        熵校准建直方图用的桶数（TensorRT 默认 2048）
MSE_NUM_CANDIDATES = 100
ENTROPY_NUM_CANDIDATES = 100
ENTROPY_NUM_BINS = 2048
CALIB_SHRINK = 0.1

# entropy 校准允许掐掉的最大激活比例。
# KL 散度衡量的是「分布形状差多少」，不是「数值差多少」，所以纯 KL 最小
# 会选出掐掉 1.7% 激活的上界 —— 实测 A4 下整体精度从 86% 崩到 66%。
# 这个上限是必需的约束，不是可选项。超过它就退回 minmax。
ENTROPY_MAX_CLIP = 0.01

# 边界自检用的极小位宽，量化算子必须能处理（对称性 0 占一格，2bit 是最下界）
MIN_BITS = 2

# 混合精度（e5）的「算不算有压缩」上限：平均权重位宽必须 <= 这个值。
# ⚠️ 没有这个约束的话，「最优配置」会选到「全部保 8bit」—— 那等于没压缩，
#    精度自然最高（87.03% ≈ FP32 87.16%），选出来毫无意义。
MIXED_MAX_AVG_BITS = 5.0

# ==============================================================================
# 设备
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# 参考实现开关
# ==============================================================================
# =1 时用 reference/quant_ref.py 覆盖 src/quant.py 里的 TODO 函数。
# 第一次跑建议开着：先确认整条流水线是通的，再关掉自己去填。
USE_REFERENCE = os.environ.get("PTQLAB_REFERENCE", "1") == "1"


# ==============================================================================
# 自检
# ==============================================================================
def ensure_dirs():
    """建好结果目录与图目录。实验脚本开头调一次，别让落盘时才炸。"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    return RESULT_DIR


def dataset_dir(ds=None):
    """torchvision 数据集应该放在哪。

    ⚠️ **CIFAR-10 是历史特例**：它的数据早就躺在 paper-code/data/ 根目录下
    （子目录名是 `cifar-10-batches-py`，由 torchvision 自己解包决定），
    不能挪 —— 挪了默认组合立刻跑不起来（已实测踩过）。
    CIFAR-100 等新数据集则各自一个子目录。
    """
    ds = ds or DATASET
    if ds == "cifar10":
        return DATA_DIR
    return os.path.join(DATA_DIR, ds)


def require_assets(need_ckpt=True):
    """启动前自检：把「缺哪个文件」一次说清，别让实验跑一半才炸。

    返回 (ok, missing)。missing 是 [(名字, 路径), ...]。
    """
    need = [(("%s 数据集目录" % DATASET), dataset_dir(), "dir")]
    if need_ckpt:
        need.append(("主基线权重（50 epoch）", CKPT_MAIN, "file"))
    missing = []
    for name, path, kind in need:
        ok = os.path.isdir(path) if kind == "dir" else os.path.isfile(path)
        if not ok:
            missing.append((name, path))
    return (len(missing) == 0), missing


def print_asset_report(need_ckpt=True):
    """打印一行自检结论。返回 True 表示可以继续跑。"""
    ok, missing = require_assets(need_ckpt)
    if ok:
        print("依赖自检 : %s + %s 数据集与权重齐全" % (DATASET, ARCH))
    else:
        print("依赖自检 : 缺 %d 项" % len(missing))
        for name, path in missing:
            print("  [缺] %s : %s" % (name, path))
    return ok


def device_info():
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0)
        free = torch.cuda.mem_get_info()[0] / 1024 ** 3
        return "cuda  %s  %.2fGB (free %.2fGB)" % (p.name, p.total_memory / 1024 ** 3, free)
    return "cpu"


def describe():
    """一行说明当前配置，跑实验时打出来，避免跑错口径还不知道。"""
    return "%s / %s / %d 类 / FP32 %.1fM params" % (
        DATASET, ARCH, NUM_CLASSES, ARCH_SPECS[ARCH]["params_m"])
