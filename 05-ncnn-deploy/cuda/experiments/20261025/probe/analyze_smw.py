#!/usr/bin/env python3
"""H-090: combine the two independent 8-round batches and size the per-layer shared tile."""
import io
import json
import os
import random
import re
import statistics
import subprocess

LAB = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy"
OUT = os.path.join(LAB, "cuda", "results-h090")
PARAM = os.path.join(LAB, "models", "resnet18_cifar10_int8_old.param")
BIN_BYTES = 32 * 4            # 一个 K 行 = 32 个 lane * 4 字节
BUDGET = 8192
SHARED_PER_SM = 65536         # devprobe: sharedPerSM
THREADS_PER_SM = 1024         # devprobe: maxThreadsPerSM
BLOCK = 256


def boot(vals, iters=20000, seed=777):
    rnd = random.Random(seed)
    meds = sorted(statistics.median([vals[rnd.randrange(len(vals))] for _ in range(len(vals))])
                  for _ in range(iters))
    return meds[int(0.025 * iters)], meds[int(0.975 * iters)]


def combine():
    batches = {}
    for name in ("ab_smw_batch1.json", "ab_smw_batch2.json"):
        with io.open(os.path.join(OUT, name), encoding="utf-8") as f:
            batches[name.split("_")[2].split(".")[0]] = json.load(f)
    allg = batches["batch1"]["gpu_impr"] + batches["batch2"]["gpu_impr"]
    allw = batches["batch1"]["wall_impr"] + batches["batch2"]["wall_impr"]
    res = {
        "per_batch": {k: {"gpu_median_pct": round(v["gpu_median_pct"], 2),
                          "gpu_ci95": [round(x, 2) for x in v["gpu_ci95"]],
                          "wall_median_pct": round(v["wall_median_pct"], 2),
                          "wall_ci95": [round(x, 2) for x in v["wall_ci95"]],
                          "positive_rounds": sum(1 for x in v["gpu_impr"] if x > 0)}
                    for k, v in batches.items()},
        "combined16": {"n": len(allg),
                       "gpu_median_pct": round(statistics.median(allg), 2),
                       "gpu_ci95": [round(x, 2) for x in boot(allg)],
                       "wall_median_pct": round(statistics.median(allw), 2),
                       "wall_ci95": [round(x, 2) for x in boot(allw)],
                       "positive_rounds": sum(1 for x in allg if x > 0),
                       "off_gpu_range": [min(batches["batch1"]["off_gpu"] + batches["batch2"]["off_gpu"]),
                                         max(batches["batch1"]["off_gpu"] + batches["batch2"]["off_gpu"])],
                       "on_gpu_range": [min(batches["batch1"]["on_gpu"] + batches["batch2"]["on_gpu"]),
                                        max(batches["batch1"]["on_gpu"] + batches["batch2"]["on_gpu"])]}
    }
    res["combined16"]["decision"] = ("default_on"
                                     if (res["combined16"]["gpu_ci95"][0] > 0
                                         and res["combined16"]["gpu_median_pct"] >= 2.0)
                                     else "keep_opt_in")
    return res


def layers():
    """Parse the param, walk the net and size the shared tile for the smw path."""
    text = io.open(PARAM, encoding="utf-8", errors="replace").read().splitlines()
    convs = [l for l in text if l.startswith("Convolution")]
    # 按 param 里的 num_output/kernel/stride/weight_data_size 反推 Cin；
    # 特征图尺寸按该固定模型的实际网络拓扑（32→16→8→4→2→1）给定。
    feats = [256, 64, 64, 64, 64, 16, 16, 16, 16, 16, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1]
    rows = []
    for idx, line in enumerate(convs):
        kv = dict(t.split("=") for t in line.split()[5:] if "=" in t)
        cout = int(kv["0"])
        kw = int(kv["1"])
        kh = int(kv.get("11", kw))
        stride = int(kv.get("3", 1))
        wsize = int(kv["6"])
        cin = wsize // (cout * kh * kw)
        M = feats[idx]
        C4 = (cin + 3) // 4
        taps = kh * kw
        eligible = M >= 2
        chunk = max(1, min(taps, BUDGET // (C4 * 32 * 4)))
        smem = chunk * C4 * 32 * 4
        K = taps * C4
        gridx = (M + 7) // 8
        gridy = (cout + 31) // 32
        warps_active = min(8, M)
        rows.append({
            "layer": line.split()[1], "cin": cin, "cout": cout, "kh": kh, "kw": kw,
            "stride": stride, "M": M, "C4": C4, "taps": taps, "K": K,
            "smw": eligible, "chunk_taps": chunk if eligible else 0,
            "shared_bytes_per_block": smem if eligible else 0,
            "blocks": gridx * gridy, "active_warps_per_block": warps_active,
            "weight_load_rows_base": gridx * gridy * warps_active * K,
            "weight_load_rows_smw": (gridx * gridy * K) if eligible else gridx * gridy * warps_active * K,
            "blocks_per_sm_shared_bound": SHARED_PER_SM // max(smem, 1),
        })
    return rows


def main():
    rep = {"ab": combine(), "layers": layers()}
    tb = sum(r["weight_load_rows_base"] for r in rep["layers"])
    ts = sum(r["weight_load_rows_smw"] for r in rep["layers"])
    rep["weight_load_rows_per_image"] = {"base": tb, "smw": ts, "ratio": round(tb / ts, 2)}
    rep["max_shared_bytes_per_block"] = max(r["shared_bytes_per_block"] for r in rep["layers"])
    with io.open(os.path.join(OUT, "smw_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep["ab"], ensure_ascii=False, indent=2))
    print("max shared/block = %d B ; rows base=%d smw=%d ratio=%.2fx" %
          (rep["max_shared_bytes_per_block"], tb, ts, tb / ts))
    for r in rep["layers"]:
        print("%-46s C=%3d C4=%3d taps=%2d K=%4d M=%4d smw=%-5s chunk=%2d shared=%6d blocks=%3d actw=%d" % (
            r["layer"], r["cin"], r["C4"], r["taps"], r["K"], r["M"], r["smw"],
            r["chunk_taps"], r["shared_bytes_per_block"], r["blocks"], r["active_warps_per_block"]))


if __name__ == "__main__":
    main()
