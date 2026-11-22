"""Run many train_prune configs in parallel; skips any run whose result JSON already exists (resumable).

    python -m train.sweep --targets 0 0.9 --criteria saliency magnitude --seeds 0 1 2 3 4

target 0 is the dense reference and is run once per seed (the criterion is irrelevant there).
Each run is a separate process with its own BLAS thread cap so parallel jobs don't fight over cores.
Per-run logs go to logs/ (gitignored); results to --out.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run_name(target: float, criterion: str, seed: int) -> str:
    return f"dense_seed{seed}" if target == 0 else f"{criterion}_s{round(target * 100)}_seed{seed}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=float, nargs="+", default=[0.0, 0.9])
    ap.add_argument("--criteria", nargs="+", default=["saliency", "magnitude"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2, help="BLAS threads per job")
    ap.add_argument("--out", default="results/part3")
    ap.add_argument("--plots", action="store_true", help="also write a curve PNG per run")
    a, extra = ap.parse_known_args(argv)  # anything else is forwarded to train_prune

    runs = []
    for seed in a.seeds:
        for t in a.targets:
            for c in (["saliency"] if t == 0 else a.criteria):
                name = run_name(t, c, seed)
                if not (Path(a.out) / f"{name}.json").exists():
                    runs.append((name, ["--target", str(t), "--criterion", c, "--seed", str(seed), "--out", a.out]))
    print(f"{len(runs)} runs to do ({a.jobs} parallel x {a.threads} threads)")
    Path("logs").mkdir(exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": str(a.threads), "OPENBLAS_NUM_THREADS": str(a.threads),
           "MKL_NUM_THREADS": str(a.threads), "VECLIB_MAXIMUM_THREADS": str(a.threads)}

    def go(item):
        name, args = item
        t0 = time.perf_counter()
        cmd = [sys.executable, "-m", "train.train_prune", *args, *extra, *([] if a.plots else ["--no-plot"])]
        with open(f"logs/{name}.log", "w") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        tail = Path(f"logs/{name}.log").read_text().strip().splitlines()
        final = next((l for l in reversed(tail) if l.startswith("FINAL")), tail[-1] if tail else "")
        print(f"[{'ok' if rc == 0 else 'FAIL'}] {name} ({time.perf_counter() - t0:.0f}s) {final[:90]}", flush=True)
        return rc

    with ThreadPoolExecutor(a.jobs) as ex:
        rcs = list(ex.map(go, runs))
    if any(rcs):
        sys.exit(1)


if __name__ == "__main__":
    main()
