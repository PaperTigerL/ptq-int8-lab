#!/usr/bin/env python3
# Build results-h021/acceptance.json from single-process 2000 logs + chunk cross-check.
import glob
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results-h021")
CHUNK = 500


def parse(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"^精度 ([0-9.]+)% \((\d+)/(\d+)\)", text, re.M)
    g = re.search(r"^最优 GPU kernel\s+([0-9.]+) ms/img", text, re.M)
    w = re.search(r"^最优 含传输墙钟\s+([0-9.]+) ms/img", text, re.M)
    mem = re.search(r"显存已用 ([0-9.]+) MB", text)
    used = re.search(r"^arena_used_mb blob=([0-9.]+) work=([0-9.]+) io=([0-9.]+)", text, re.M)
    rc = re.search(r"^exit_code=(\d+)$", text, re.M)
    if not (m and g):
        return None
    return {
        "acc": float(m.group(1)),
        "correct": int(m.group(2)),
        "n": int(m.group(3)),
        "gpu_ms_per_img": float(g.group(1)),
        "wall_ms_per_img": float(w.group(1)) if w else None,
        "gpu_mem_mb": float(mem.group(1)) if mem else None,
        "blob_arena_used_mb": float(used.group(1)) if used else None,
        "work_arena_used_mb": float(used.group(2)) if used else None,
        "exit_code": int(rc.group(1)) if rc else None,
    }


def cpu_baseline(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    t = open(p, encoding="utf-8", errors="replace").read()
    m = re.search(r"acc ([0-9.]+)% \((\d+)/(\d+)\) \| ([0-9.]+) ms/img", t)
    if not m:
        return None
    return {"acc": float(m.group(1)), "correct": int(m.group(2)),
            "n": int(m.group(3)), "ms_per_img": float(m.group(4)),
            "log": "results-h021/" + name}


cpu = {}
for key, name in (("int8", "cpu_int8_2000.txt"), ("fp32", "cpu_fp32_2000.txt")):
    v = cpu_baseline(name)
    if v:
        cpu[key] = v

single = []
for path in sorted(glob.glob(os.path.join(OUT, "single_*.log"))):
    name = os.path.basename(path)[:-4]
    m = re.match(r"^single_(default|fallback)_b(64|256)$", name)
    if not m:
        continue
    rec = parse(path)
    if rec is None:
        continue
    rec.update({"path": m.group(1), "batch": int(m.group(2)),
                "log": "results-h021/" + os.path.basename(path)})
    single.append(rec)

chunks = []
for path in sorted(glob.glob(os.path.join(OUT, "*_c*_p*.log"))):
    name = os.path.basename(path)[:-4]
    m = re.match(r"^(default|fallback)_b(64|256)_c(\d)_p(\d)$", name)
    if not m:
        continue
    rec = parse(path)
    if rec is None:
        continue
    rec.update({"path": m.group(1), "batch": int(m.group(2)),
                "chunk": int(m.group(3)), "pass": int(m.group(4)),
                "log": "results-h021/" + os.path.basename(path)})
    chunks.append(rec)

chunk_agg = []
for path in ("default", "fallback"):
    for batch in (64, 256):
        for p in sorted({r["pass"] for r in chunks}):
            rs = [r for r in chunks if r["path"] == path and r["batch"] == batch and r["pass"] == p]
            if len(rs) != 4:
                continue
            rs.sort(key=lambda r: r["chunk"])
            correct = sum(r["correct"] for r in rs)
            n = sum(r["n"] for r in rs)
            gpu_ms = sum(r["gpu_ms_per_img"] * r["n"] for r in rs) / n
            chunk_agg.append({
                "path": path, "batch": batch, "pass": p, "n": n,
                "correct": correct, "acc": round(100.0 * correct / n, 4),
                "gpu_ms_per_img_mean": round(gpu_ms, 4),
                "gpu_ms_per_img_min": min(r["gpu_ms_per_img"] for r in rs),
                "gpu_ms_per_img_max": max(r["gpu_ms_per_img"] for r in rs),
                "chunks": [{"chunk": r["chunk"], "correct": r["correct"],
                            "gpu_ms_per_img": r["gpu_ms_per_img"],
                            "log": r["log"], "exit_code": r["exit_code"]} for r in rs],
            })

cpu_int8 = cpu.get("int8", {}).get("acc")
for r in single:
    if cpu_int8 is not None:
        r["delta_vs_cpu_int8_pp"] = round(r["acc"] - cpu_int8, 4)

p1 = [r for r in chunk_agg if r["pass"] == 1]
chunk_max_delta = max((abs(r["acc"] - cpu_int8) for r in p1), default=None) if cpu_int8 is not None else None
single_max_delta = max((abs(r["delta_vs_cpu_int8_pp"]) for r in single), default=None)

doc = {
    "schema": "ncnn-cuda/acceptance@1",
    "task": "H-20261023-021",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "model": "resnet18_cifar10_int8_old",
    "data": "CIFAR-10 test_batch.raw 前 2000 张（与 CPU int8 基线同批）",
    "device": "GTX 1650 Ti / WSL CUDA 12.4",
    "fork_commit": "8918acf",
    "fork_commits": ["4cc02aa", "fc73aa4", "1aa392c", "b1a94f0", "8918acf"],
    "build_dir": "/home/user/ncnn-cuda/build-ptq",
    "harness": {
        "source": "05-ncnn-deploy/cuda/ncnn_cuda_eval_2000c.cpp",
        "base": "ncnn_cuda_eval_h015.cpp（release-before-reset；reset 在有活块时自动 defer）",
        "single_process": "blob arena 1280MB 单调增长、地址全程不复用；"
                          "work arena 64MB 每图回退（实测 work 用量 3.8MB）；batch 64/256",
        "chunked_cross_check": "前 2000 张拆成 4 个 500 张、每个独立进程 --no-reset；两轮",
        "why": "该 fork 的 blob arena 一旦跨图复用地址就会读回上一张的数据（见 reset_investigation）。"
               "两种口径都保证地址不复用，结果一致。",
    },
    "cpu": cpu,
    "single_process_runs": single,
    "chunked_cross_check": chunk_agg,
    "acceptance": {
        "threshold_pp": 0.3,
        "metric": "GPU int8 top-1 vs CPU int8 top-1（同模型、同 2000 张）",
        "max_abs_delta_single_pp": single_max_delta,
        "max_abs_delta_chunk_pp": chunk_max_delta,
        "pass": single_max_delta is not None and single_max_delta <= 0.3,
    },
    "reset_investigation": {
        "evidence": "results-h021/reset_alias_evidence.md",
        "summary": "blob arena 每图 reset 复用时，20 张准确率 55%（no-reset 100%）；"
                   "reset_every=2/4/8 分别为 80/90/100%；只 reset workspace 时 100%。"
                   "首个分叉出现在第一次地址复用后的第 2 张图（logits FNV 哈希不同）。"
                   "deferred=0、stale=0，不是陈旧句柄释放；复用地址被未跟踪的读者读到。"
                   "验收因此改为单调 blob arena，绝不 reset。",
        "reset_every_sweep": {"1": 55.0, "2": 80.0, "4": 90.0, "8": 100.0,
                              "16": 100.0, "no_reset": 100.0},
    },
}
path = os.path.join(OUT, "acceptance.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(doc, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(path)
for r in single:
    print("single {path} b{batch}: {correct}/{n} = {acc}%  delta {delta:+.4f} pp  "
          "gpu {gpu:.4f} ms/img  wall {wall:.4f}  blob {blob} MB".format(
              delta=r.get("delta_vs_cpu_int8_pp", float("nan")),
              gpu=r["gpu_ms_per_img"], wall=r["wall_ms_per_img"] or 0,
              blob=r["blob_arena_used_mb"], **r))
for r in p1:
    print("chunk p1 {path} b{batch}: {correct}/{n} = {acc}%  gpu {gpu:.4f} ms/img "
          "[{mn:.3f},{mx:.3f}]".format(gpu=r["gpu_ms_per_img_mean"],
                                       mn=r["gpu_ms_per_img_min"],
                                       mx=r["gpu_ms_per_img_max"], **r))
print("cpu:", cpu)
print("acceptance pass:", doc["acceptance"]["pass"],
      "single_max_delta:", single_max_delta, "chunk_max_delta:", chunk_max_delta)
