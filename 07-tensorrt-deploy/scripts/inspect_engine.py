"""把 engine 的逐层 tactic 打出来，作为「到底有没有跑目标精度」的证据。

不要凭 plan 大小或跑分猜精度——本节目的实现踩过一次：
  fp16 engine 和 fp32 engine 的 plan 都是 44.9 MB，跑分也只差 1.28 倍，
  看起来「fp16 生效了」，但逐层一看全是 volta_scudnn_*（fp32 kernel）——
  真正生效要靠 tactic 名验证。

用法:
    python inspect_engine.py <plan.trt> [batch]
"""
import json
import sys
from collections import Counter
from pathlib import Path

import tensorrt as trt


def main():
    plan_path = Path(sys.argv[1])
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 3072

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    with open(plan_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    ctx.set_input_shape("input", (batch, 3, 32, 32))

    insp = engine.create_engine_inspector()
    insp.execution_context = ctx
    info = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
    layers = info["Layers"]
    print(f"engine={plan_path.name}  layers={len(layers)}  batch={batch}")

    c = Counter()
    for L in layers:
        tl = L.get("TacticName", "").lower()
        kinds = set()
        for tok in ("int8", "i8", "fp16", "half", "f16", "scudnn", "xmma", "turing", "volta"):
            if tok in tl:
                kinds.add(tok)
        c[tuple(sorted(kinds)) or ("(无 tactic)",)] += 1
    print("\ntactic 关键字分布:")
    for k, v in c.most_common():
        print(f"  {v:3d}  {k}")

    print("\n逐层 tactic:")
    for L in layers:
        print(f"  {L.get('Name', '')[:50]:50s} {L.get('TacticName', '')}")


if __name__ == "__main__":
    main()
