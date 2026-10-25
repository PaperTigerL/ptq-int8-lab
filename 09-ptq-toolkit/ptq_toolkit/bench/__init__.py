"""基准采集（M1 只引用已有结果，不重测 GPU）。"""
from .collect import collect

__all__ = ["collect"]
