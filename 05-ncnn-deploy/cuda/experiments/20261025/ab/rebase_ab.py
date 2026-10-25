#!/usr/bin/env python3
import io
p = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/ab_graph_default_on.py"
t = io.open(p, encoding="utf-8").read()
t = t.replace('BIN = os.path.join(LAB, "ncnn_cuda_graph_eval")',
              'BIN_OFF = os.path.join(LAB, "ncnn_cuda_eval_2000c_lazyw8")   # 8959d4c 普通路径\n'
              'BIN_ON = os.path.join(LAB, "ncnn_cuda_graph_eval")            # graph 默认开')
t = t.replace('''    if cand:
        e["LD_PRELOAD"] = SHIM            # default-on; env NCNN_CUDA_GRAPH unset
    else:
        e["NCNN_CUDA_GRAPH"] = "0"        # explicit off''',
              '''    if cand:
        e["LD_PRELOAD"] = SHIM            # graph 默认开（env NCNN_CUDA_GRAPH 未设）
    # off = 8959d4c 普通二进制，环境保持干净''')
t = t.replace('    cmd = [BIN, P, B, D,', '    cmd = [BIN_ON if cand else BIN_OFF, P, B, D,')
t = t.replace('json.dump({"binary": BIN,', 'json.dump({"binary_off": BIN_OFF, "binary_on": BIN_ON,')
io.open(p, "w", encoding="utf-8", newline="\n").write(t)
print("ab script rebased")
