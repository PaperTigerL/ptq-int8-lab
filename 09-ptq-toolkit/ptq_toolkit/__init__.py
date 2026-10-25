"""PTQ 工具链：PyTorch 模型 + 校准数据 → 量化 → 导出 → 基准 → 报告。

M1 只支持 ResNet18-CIFAR10。多模型（W4A8）、TRT 对比、CI 属后续里程碑。

依赖 `../03-ptq-lab`（量化语义的 ground truth），不重复实现 BN 折叠、
校准与精度评测 —— 重复实现会导致精度口径与 03/06/07 不一致。
"""
__version__ = "0.1.0"
__all__ = ["pipeline", "cli"]
