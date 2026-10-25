"""模型 registry。

⭐ 为什么用 registry 而不是 pipeline 里 if/else：
   「加模型 = 加一条 loader 条目」是 M2 的验收判据之一。
   if/else 的写法在加第三、第四个模型时会立刻失控，
   而且 pipeline 会同时负责「选模型」和「跑流程」两件事。
"""
from .registry import (REGISTRY, ModelSpec, get_model, list_models, topo_kind,
                        topo_plan)

__all__ = ["REGISTRY", "ModelSpec", "get_model", "list_models", "topo_kind"]
