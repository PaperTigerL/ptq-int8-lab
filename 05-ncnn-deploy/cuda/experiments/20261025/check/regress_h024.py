#!/usr/bin/env python3
"""H-20261024-004 single-command regression for ncnn CUDA implicit int8.

Runs 2000 CIFAR-10 images x 4 configs:
  default implicit / NCNN_CUDA_IMPLICIT=0 fallback  x  batch 64/256

Compares against baseline (H-20261023-021):
  correct == 1739/2000 (86.95%) for every config          -> hard failure
  implicit < 6.0 ms/img and fallback > 8.0 ms/img         -> path/catastrophic check
  absolute deltas vs baseline are recorded in the JSON

A tight absolute ms gate is not reproducible on this WSL laptop GPU: the same
binary measured 2.65-4.06 ms/img across runs (clocks 1350-2100 MHz).  We
therefore burn in one discarded 2000-image pass, gate accuracy exactly, and use
a generous 6.0 ms ceiling to catch real regressions (e.g. silently taking the
fallback path) without failing on clock state.

Exit code 0 = PASS, 1 = drift/failure, 2 = usage/setup error.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = "/mnt/d/WorkBuddy/ShowBro/lab/05-ncnn-deploy"
BIN = os.path.join(HERE, "ncnn_cuda_eval_2000c")
MODEL_DIR = os.path.join(LAB, "models")
PARAM = os.path.join(MODEL_DIR, "resnet18_cifar10_int8_old.param")
BMODEL = os.path.join(MODEL_DIR, "resnet18_cifar10_int8_old.bin")
DATA = os.path.join(LAB, "data_raw", "test_batch.raw")

# Baseline H-20261023-021 (results-h021/acceptance.json, fork 8918acf).
BASELINE = {
    "schema": "ncnn-cuda/implicit-int8-baseline@1",
    "task": "H-20261023-021",
    "fork_commit": "8918acf",
    "data": "CIFAR-10 test_batch.raw first 2000 (same batch as CPU int8)",
    "n": 2000,
    "correct": 1739,
    "acc_pct": 86.95,
    "cpu_int8_acc_pct": 86.95,
    "gpu_ms_per_img": {
        "default_b64": 3.9259,
        "default_b256": 3.8879,
        "fallback_b64": 15.9404,
        "fallback_b256": 16.074,
    },
    "implicit_band_ms": [3.89, 3.93],
    "fallback_band_ms": [15.94, 16.07],
    "accuracy_tolerance_pp": 0.0,
    "source": "D:/WorkBuddy/ShowBro/lab/05-ncnn-deploy/cuda/results-h021/acceptance.json",
}

RE_ACC = re.compile(r"^精度 (\d+\.\d+)% \((\d+)/(\d+)\)", re.M)
RE_GPU = re.compile(r"^最优 GPU kernel\s+([\d.]+) ms/img", re.M)
RE_WALL = re.compile(r"^最优 含传输墙钟\s+([\d.]+) ms/img", re.M)
RE_TRANSFER = re.compile(r"H2D ([\d.]+) \+ D2H ([\d.]+)\)")


def run_one(path, batch, limit, warmup, timeout):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:/home/user/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    if path == "fallback":
        env["NCNN_CUDA_IMPLICIT"] = "0"
    else:
        env.pop("NCNN_CUDA_IMPLICIT", None)
    cmd = [BIN, PARAM, BMODEL, DATA, "--int8", "--limit", str(limit),
           "--batch", str(batch), "--repeat", "1", "--warmup", str(warmup),
           "--reset-every", "1"]
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    text = proc.stdout.decode("utf-8", "replace")
    return proc.returncode, text, time.time() - t0


def parse(text):
    out = {}
    m = RE_ACC.search(text)
    if m:
        out["acc_pct"] = float(m.group(1))
        out["correct"] = int(m.group(2))
        out["n"] = int(m.group(3))
    m = RE_GPU.search(text)
    if m:
        out["gpu_ms_per_img"] = float(m.group(1))
    m = RE_WALL.search(text)
    if m:
        out["wall_ms_per_img"] = float(m.group(1))
    m = RE_TRANSFER.search(text)
    if m:
        out["h2d_ms_per_img"] = float(m.group(1))
        out["d2h_ms_per_img"] = float(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(HERE, "results-h024"))
    ap.add_argument("--burn-in", type=int, default=1,
                    help="discarded 2000-image passes before measuring (clock ramp)")
    ap.add_argument("--strict-perf", action="store_true",
                    help="also fail when a config is >10%% slower than the H-021 baseline")
    args = ap.parse_args()

    for p in (BIN, PARAM, BMODEL, DATA):
        if not os.path.exists(p):
            print("missing required file: %s" % p, file=sys.stderr)
            return 2

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.out, stamp)
    os.makedirs(run_dir, exist_ok=True)
    full = args.limit >= BASELINE["n"]
    result = {
        "schema": "ncnn-cuda/implicit-int8-regress@1",
        "task": "T-20261024-04 / H-20261024-004",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": "GTX 1650 Ti / WSL CUDA 12.4",
        "binary": BIN,
        "limit": args.limit,
        "full_2000": full,
        "strict_perf": args.strict_perf,
        "baseline": BASELINE,
        "thresholds": {"accuracy": "exact 1739/2000 for full runs",
                       "default_ceiling_ms": 6.0,
                       "fallback_floor_ms": 8.0,
                       "fallback_ceiling_ms": 25.0,
                       "accuracy_tolerance_pp": 0.0},
        "configs": [],
        "drift": [],
    }
    try:
        repo = "/home/user/ncnn-cuda"
        result["fork_commit"] = subprocess.run(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10
        ).stdout.decode().strip()
        dirty = subprocess.run(
            ["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10
        ).stdout.decode().strip()
        result["fork_dirty_tracked"] = bool(dirty)
    except Exception as e:  # pragma: no cover - best effort metadata
        result["fork_commit"] = "unknown"
        result["fork_commit_error"] = str(e)

    print("run dir: %s" % run_dir)
    if args.burn_in > 0 and full:
        for b in range(args.burn_in):
            print("[burn-in %d/%d] default_b64 2000 images (discarded)" % (b + 1, args.burn_in), flush=True)
            rcb, textb, secsb = run_one("default", 64, args.limit, args.warmup, timeout=1800)
            with open(os.path.join(run_dir, "burnin_default_b64_%d.log" % (b + 1)), "w", encoding="utf-8") as f:
                f.write(textb)
                f.write("exit_code=%d\n" % rcb)
            mb = parse(textb)
            result.setdefault("burn_in", []).append(
                {"rc": rcb, "gpu_ms_per_img": mb.get("gpu_ms_per_img"), "elapsed_s": round(secsb, 2)})
    for path in ("default", "fallback"):
        for batch in (64, 256):
            tag = "%s_b%d" % (path, batch)
            print("[run] %-12s limit=%d batch=%d ..." % (tag, args.limit, batch), flush=True)
            rc, text, secs = run_one(path, batch, args.limit, args.warmup, timeout=1800)
            log = os.path.join(run_dir, "%s.log" % tag)
            with open(log, "w", encoding="utf-8") as f:
                f.write(text)
                f.write("exit_code=%d\n" % rc)
            m = parse(text)
            cfg = {"tag": tag, "path": path, "batch": batch, "exit_code": rc,
                   "elapsed_s": round(secs, 2), "log": log}
            cfg.update(m)
            result["configs"].append(cfg)
            print("      rc=%d acc=%s correct=%s gpu=%s ms/img wall=%s" % (
                rc, m.get("acc_pct"), m.get("correct"), m.get("gpu_ms_per_img"),
                m.get("wall_ms_per_img")), flush=True)

    # ---- checks ----
    def fail(msg):
        result["drift"].append(msg)

    for cfg in result["configs"]:
        tag = cfg["tag"]
        if cfg["exit_code"] != 0:
            fail("%s: exit_code=%d" % (tag, cfg["exit_code"]))
        if "acc_pct" not in cfg or "gpu_ms_per_img" not in cfg:
            fail("%s: log parse incomplete" % tag)
            continue
        if full:
            if cfg["correct"] != BASELINE["correct"]:
                fail("%s: correct %d != baseline %d (acc %.2f%% vs %.2f%%, delta %.2f pp)" % (
                    tag, cfg["correct"], BASELINE["correct"], cfg["acc_pct"],
                    BASELINE["acc_pct"], cfg["acc_pct"] - BASELINE["acc_pct"]))
        # Partial (<2000) runs are smoke checks only: short runs sit in a lower
        # GPU clock state, so per-image timing is not comparable to baseline.
        if not full:
            if tag.startswith("default") and cfg["gpu_ms_per_img"] > 8.0:
                fail("%s: implicit path looks like fallback (%.4f ms/img)" % (tag, cfg["gpu_ms_per_img"]))
            if tag.startswith("fallback") and cfg["gpu_ms_per_img"] < 8.0:
                fail("%s: fallback switch ignored (%.4f ms/img)" % (tag, cfg["gpu_ms_per_img"]))
            continue
        base = BASELINE["gpu_ms_per_img"][tag]
        cfg["baseline_gpu_ms_per_img"] = base
        delta = (cfg["gpu_ms_per_img"] / base - 1.0) * 100.0
        cfg["delta_vs_baseline_pct"] = round(delta, 2)
        if tag.startswith("default"):
            cfg["hard_ceiling_ms"] = 6.0
            if cfg["gpu_ms_per_img"] > 6.0:
                fail("%s: gpu %.4f ms/img > 6.0 hard ceiling (baseline %.4f, %+.1f%%)" % (
                    tag, cfg["gpu_ms_per_img"], base, delta))
        else:
            cfg["hard_floor_ms"] = 8.0
            cfg["hard_ceiling_ms"] = 25.0
            if cfg["gpu_ms_per_img"] < 8.0:
                fail("%s: fallback switch ignored (%.4f ms/img)" % (tag, cfg["gpu_ms_per_img"]))
            if cfg["gpu_ms_per_img"] > 25.0:
                fail("%s: fallback gpu %.4f ms/img > 25.0 hard ceiling" % (tag, cfg["gpu_ms_per_img"]))
        if delta > 10.0:
            result.setdefault("warnings", []).append(
                "%s: %+.1f%% vs baseline (still below hard ceiling; WSL GPU clock state)" % (tag, delta))
        if args.strict_perf and delta > 10.0:
            fail("%s: gpu %.4f ms/img is %+.1f%% vs baseline %.4f (strict-perf >10%%)" % (
                tag, cfg["gpu_ms_per_img"], delta, base))

    result["pass"] = len(result["drift"]) == 0
    result["verdict"] = "PASS" if result["pass"] else "DRIFT"
    out_json = os.path.join(run_dir, "regress.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\nverdict: %s" % result["verdict"])
    if result["drift"]:
        for d in result["drift"]:
            print("  DRIFT: %s" % d)
    print("json: %s" % out_json)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
