#!/usr/bin/env python3
"""H-061: cumulative accounting 644efc3 -> HEAD plus per-optimization toggles."""
import json
import os
import random
import re
import statistics
import subprocess

LAB = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda"
B644 = os.path.join(LAB, "ncnn_cuda_eval_2000c_644efc3")
BHEAD = os.path.join(LAB, "ncnn_cuda_eval_2000c_lazyw8")   # 8959d4c
BGRAPH = os.path.join(LAB, "ncnn_cuda_graph_eval")
SHIM = os.path.join(LAB, "graph_shim.so")
P = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/models/resnet18_cifar10_int8_old.param"
B = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/models/resnet18_cifar10_int8_old.bin"
D = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy/data_raw/test_batch.raw"
OUT = os.path.join(LAB, "results-h061")
RE_ACC = re.compile(r"^精度 (\d+\.\d+)% \((\d+)/(\d+)\)", re.M)
RE_GPU = re.compile(r"^最优 GPU kernel\s+([\d.]+)", re.M)
RE_WALL = re.compile(r"^最优 含传输墙钟\s+([\d.]+)", re.M)


def clean_env(extra):
    e = dict(os.environ)
    e["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:/home/user/cuda/lib64"
    for k in list(e):
        if k.startswith("NCNN_CUDA_") or k == "LD_PRELOAD":
            e.pop(k, None)
    e.update(extra)
    return e


def run(binary, extra):
    cmd = [binary, P, B, D, "--int8", "--limit", "512", "--batch", "64",
           "--repeat", "1", "--warmup", "30"]
    text = subprocess.run(cmd, env=clean_env(extra), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=900).stdout.decode("utf-8", "replace")
    acc = RE_ACC.search(text); gpu = RE_GPU.search(text); wall = RE_WALL.search(text)
    if not acc or not gpu or not wall:
        raise SystemExit("run failed:\n" + text[-600:])
    return float(gpu.group(1)), float(wall.group(1)), int(acc.group(2)) / int(acc.group(3)) * 100.0


def boot(vals, iters=10000, seed=12345):
    rnd = random.Random(seed)
    meds = sorted(statistics.median([vals[rnd.randrange(len(vals))] for _ in range(len(vals))]) for _ in range(iters))
    return meds[int(0.025 * iters)], meds[int(0.975 * iters)]


def pair(name, a, b, rounds, a_args, b_args):
    """a = off/baseline, b = on/candidate; interleaved, alternating order."""
    av, bv, aw, bw = [], [], [], []
    for r in range(rounds):
        order = ("a", "b") if r % 2 == 0 else ("b", "a")
        v = {}
        for which in order:
            g, w, acc = run(a if which == "a" else b,
                            a_args if which == "a" else b_args)
            v[which] = (g, w)
            print("%s r%d %s: gpu %.4f wall %.4f acc %.2f%%" % (name, r + 1, which, g, w, acc), flush=True)
        av.append(v["a"][0]); bv.append(v["b"][0])
        aw.append(v["a"][1]); bw.append(v["b"][1])
    gi = [(x - y) / x * 100 for x, y in zip(av, bv)]
    wi = [(x - y) / x * 100 for x, y in zip(aw, bw)]
    gmed = statistics.median(gi); gci = boot(gi)
    wmed = statistics.median(wi); wci = boot(wi)
    res = {"name": name, "rounds": rounds,
           "a_gpu": av, "b_gpu": bv, "a_wall": aw, "b_wall": bw,
           "a_gpu_med": statistics.median(av), "b_gpu_med": statistics.median(bv),
           "a_wall_med": statistics.median(aw), "b_wall_med": statistics.median(bw),
           "gpu_med_pct": gmed, "gpu_ci95": list(gci),
           "wall_med_pct": wmed, "wall_ci95": list(wci)}
    print("SUMMARY %s gpu %.4f->%.4f %+.2f%% ci[%.2f,%.2f] wall %+.2f%% ci[%.2f,%.2f]" % (
        name, res["a_gpu_med"], res["b_gpu_med"], gmed, gci[0], gci[1],
        wmed, wci[0], wci[1]), flush=True)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {}
    out["cumulative"] = pair("644efc3_vs_HEAD", B644, BHEAD, 8, {}, {})
    out["graph"] = pair("graph_off_vs_on", BHEAD, BGRAPH, 4,
                        {"NCNN_CUDA_GRAPH": "0"}, {"LD_PRELOAD": SHIM})
    out["maxpool"] = pair("maxpool_off_vs_on", BHEAD, BHEAD, 4,
                          {"NCNN_CUDA_POOL_FUSE_PAD": "0"}, {})
    out["gemv"] = pair("gemv_off_vs_on", BHEAD, BHEAD, 4,
                       {"NCNN_CUDA_IP_GEMV": "0"}, {})
    with open(os.path.join(OUT, "accounting.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
