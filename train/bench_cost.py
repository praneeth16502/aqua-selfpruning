"""Part 4: real cost measurement of pruned models.

    OMP_NUM_THREADS=1 python -m train.bench_cost --models results/part4/models/*.npz

For each saved model (W, b, mask per layer) compares three inference paths on the MNIST test set:
    dense      x @ where(mask, W, 0)    the dense-times-zeros shortcut: same work as an unpruned model
    compact    compaction (dead inputs/units removed) + dense BLAS on the smaller matrices
    csr        compaction + our CSR kernel (work ~ live weights only)
Checks all three give the same predictions and the same test accuracy, then times them at several batch
sizes (median of repeats, paths interleaved round-robin so machine drift affects all paths equally).  Run on an otherwise idle machine with a fixed BLAS thread count: timings taken while other
jobs run are not comparable.  Writes <out>/cost.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import time
from pathlib import Path

import numpy as np

from nn.sparse import SparseMLP
from train.data import get_dataset


def load_model(path: Path):
    z = np.load(path)
    n = len([k for k in z.files if k.startswith("W")])
    return [z[f"W{i}"] for i in range(n)], [z[f"mask{i}"] for i in range(n)], [z[f"b{i}"] for i in range(n)]


def dense_forward(x, Weffs, bs):
    """Plain dense forward on pre-masked weights: the dense-times-zeros shortcut, nothing else."""
    h = x
    for i, (W, b) in enumerate(zip(Weffs, bs)):
        h = h @ W + b
        if i < len(Weffs) - 1:
            h = np.maximum(h, 0)
    return h


def _calibrate(fn, x, block_time: float) -> int:
    """Number of calls so that one timing block takes at least `block_time` seconds."""
    fn(x)  # warm up
    n = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            fn(x)
        if time.perf_counter() - t0 >= block_time:
            return n
        n *= 2


def time_interleaved(paths: dict, x, repeats: int, block_time: float = 0.03) -> dict:
    """Median seconds per call for each path, measured ROUND-ROBIN: every repeat times each path once, back to back.
    The machine's speed drifts (we saw +-20% on identical work between models), and interleaving makes that drift hit
    every path equally, so the RATIOS are stable even when absolute times move."""
    n = {k: _calibrate(f, x, block_time) for k, f in paths.items()}
    samples = {k: [] for k in paths}
    for _ in range(repeats):
        for k, f in paths.items():
            t0 = time.perf_counter()
            for _ in range(n[k]):
                f(x)
            samples[k].append((time.perf_counter() - t0) / n[k])
    return {k: float(np.median(v)) for k, v in samples.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 32, 1000])
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--warmup-seconds", type=float, default=3.0)
    ap.add_argument("--out", default="results/part4")
    a = ap.parse_args(argv)

    data = get_dataset("mnist", dtype=np.float32)
    x_all, y_all = data.x_test, data.y_test
    # Warm the CPU up before timing anything: right after training (or with the machine idle at low clock) the first
    # model timed came out ~10% off, and the 0%-sparsity sanity row (dense vs compact on the SAME network, must be
    # ~1.00x) showed 1.13x.  A few seconds of matmul first makes that row stable.
    w = np.random.default_rng(0).standard_normal((784, 256)).astype(np.float32)
    t_end = time.perf_counter() + a.warmup_seconds
    while time.perf_counter() < t_end:
        x_all[:1000] @ w
    rows = []
    for path in sorted(map(Path, a.models)):
        Ws, masks, bs = load_model(path)
        s = SparseMLP(Ws, masks, bs)
        # Pre-mask ONCE: timing must be the matmul, not a per-call np.where copy (an earlier version did that and
        # the 0%-sparsity row exposed it: "compact" looked 5x faster than dense on an identical network).
        Weff = [np.where(m, W, 0) for W, m in zip(Ws, masks)]
        paths = {"dense": lambda x: dense_forward(x, Weff, bs),
                 "compact": s.forward_compact_dense, "csr": s.forward_csr}
        preds = {k: f(x_all).argmax(1) for k, f in paths.items()}
        acc = {k: float((p == y_all).mean()) for k, p in preds.items()}
        agree = all(np.array_equal(preds["dense"], p) for p in preds.values())
        sparsity = 1 - sum(int(m.sum()) for m in masks) / sum(m.size for m in masks)
        rec = {"model": path.name, "sparsity": sparsity, "test_acc": acc, "predictions_identical": agree,
               **s.cost(), "seconds_per_batch": {}}
        for bsz in a.batches:
            xb = np.ascontiguousarray(x_all[:bsz])
            # dense and compact are interleaved with each other only. CSR is timed on its own afterwards: it allocates
            # an (N x nnz) temporary (~1 GB at batch 1000, 0% sparsity) that evicts the caches, and when it sat in
            # the same round-robin, the path timed right after it (dense) ran cold and the 0% row read 1.20x.
            t = time_interleaved({"dense": paths["dense"], "compact": paths["compact"]}, xb, a.repeats)
            t.update(time_interleaved({"csr": paths["csr"]}, xb, max(3, a.repeats // 3)))
            rec["seconds_per_batch"][str(bsz)] = t
        rows.append(rec)
        t = rec["seconds_per_batch"]
        print(f"{path.name:<28} sparsity {sparsity:6.2%}  acc {acc['dense']:.4f} same={agree}  "
              f"MACs dense {rec['macs_per_sample']['dense']:,} compact {rec['macs_per_sample']['compact_dense']:,} "
              f"csr {rec['macs_per_sample']['csr']:,}")
        for bsz in a.batches:
            d = t[str(bsz)]
            print(f"    batch {bsz:>5}: dense {1e3*d['dense']:8.3f} ms | compact {1e3*d['compact']:8.3f} ms "
                  f"({d['dense']/d['compact']:.2f}x) | csr {1e3*d['csr']:8.3f} ms ({d['dense']/d['csr']:.2f}x)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"machine": platform.platform(), "processor": platform.processor(), "numpy": np.__version__,
            "note": "single process; BLAS threads as set by the environment (OMP_NUM_THREADS etc.)"}
    (out / "cost.json").write_text(json.dumps({"meta": meta, "models": rows}, indent=2))
    print(f"wrote {out}/cost.json")


if __name__ == "__main__":
    main()
