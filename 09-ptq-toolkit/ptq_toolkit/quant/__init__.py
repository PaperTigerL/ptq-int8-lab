"""PTQ 量化核心（per-channel / per-group W8/W4，对称码空间）。"""
from .ptq import (ActQuantObserver, ActQuantSpec, PTQResult, QuantizedTensor,
                  act_spec_from_range, qmax_of, quantize_per_channel_sym,
                  quantize_per_group_sym, quantize_model, quantize_weight_sym,
                  round_half_away)

__all__ = ["ActQuantObserver", "ActQuantSpec", "PTQResult", "QuantizedTensor",
           "act_spec_from_range", "qmax_of", "quantize_per_channel_sym",
           "quantize_per_group_sym", "quantize_model", "quantize_weight_sym",
           "round_half_away"]
