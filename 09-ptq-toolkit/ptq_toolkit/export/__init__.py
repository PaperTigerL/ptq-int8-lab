"""权重 / 拓扑导出（格式对齐 06-gpu-int8 的 PTQW）。

PTQC 格式层（ptqc）只依赖标准库，无 torch 环境也能跑；
依赖 torch 的 exporter 符号走 PEP 562 惰性导入，避免 `ptq ptqc` CLI
在无 torch 机器上仅仅因为 import 本包就失败。
"""
from .ptqc import (conv_names, ranges_from_store, read_ptqc,
                   weight_conv_count, write_ptqc, write_ptqc_for_model)

__all__ = ["conv_names", "ranges_from_store", "read_ptqc",
           "weight_conv_count", "write_ptqc", "write_ptqc_for_model",
           "Block", "ConvRecord", "export_weights", "extract_topology",
           "replay_topology", "write_quant_config"]

_LAZY = {
    "Block": "exporter",
    "ConvRecord": "exporter",
    "export_weights": "exporter",
    "extract_topology": "exporter",
    "replay_topology": "exporter",
    "write_quant_config": "exporter",
}


def __getattr__(name):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{mod}", __name__), name)
