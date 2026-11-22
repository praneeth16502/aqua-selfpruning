"""Part 4: real cost measurement of pruned models.

    OMP_NUM_THREADS=1 python -m train.bench_cost --models results/part4/models/*.npz

For each saved model (W, b, mask per layer) compares three inference paths on the MNIST test set:
    dense      x @ where(mask, W, 0)    the dense-times-zeros shortcut: same work as an unpruned model
    compact    compaction (dead inputs/units removed) + dense BLAS on the smaller matrices
    csr        compaction + our CSR kernel (work ~ live weights only)
Checks all three give the same predictions and the same test accuracy, then times them (median of repeats) at
several batch sizes.  Run on an otherwise idle machine with a fixed BLAS thread count: timings taken while other
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


def dense_forward(x, Ws, masks, bs):
    h = x
    for i, (W, m, b) in enumerate(zip(Ws, masks, bs)):
        h = h @ np.where(m, W, 0) + b
        if i < len(Ws) - 1:
            h = np.maximum(h, 0)
    return h


def timeit(fn, x, repeats: int, min_time: float = 0.2) -> float:
    """Median seconds per call over `repeats` timing blocks (each block loops until >= min_time/repeats)."""
    fn(x)  # warm up
    n = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            fn(x)
        dt = time.perf_counter() - t0
        if dt >= min_time / repeats:
            break
        n *= 2
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n):
            fn(x)
        samples.append((time.perf_counter() - t0) / n)
    return float(np.median(samples))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 32, 1000])
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--out", default="results/part4")
    a = ap.parse_args(argv)

    data = get_dataset("mnist", dtype=np.float32)
    x_all, y_all = data.x_test, data.y_test
    rows = []
    for path in sorted(map(Path, a.models)):
        Ws, masks, bs = load_model(path)
        s = SparseMLP(Ws, masks, bs)
        # Pre-mask once for the dense path so its timing is the matmul, not np.where.
        Weff = [np.where(m, W, 0) for W, m in zip(Ws, masks)]
        ones = [np.ones_like(m) for m in masks]
        paths = {"dense": lambda x: dense_forward(x, Weff, ones, bs),
                 "compact": s.forward_compact_dense, "csr": s.forward_csr}
        preds = {k: f(x_all).argmax(1) for k, f in paths.items()}
        acc = {k: float((p == y_all).mean()) for k, p in preds.items()}
        agree = all(np.array_equal(preds["dense"], p) for p in preds.values())
        sparsity = 1 - sum(int(m.sum()) for m in masks) / sum(m.size for m in masks)
        rec = {"model": path.name, "sparsity": sparsity, "test_acc": acc, "predictions_identical": agree,
               **s.cost(), "seconds_per_batch": {}}
        for bsz in a.batches:
            xb = np.ascontiguousarray(x_all[:bsz])
            rec["seconds_per_batch"][str(bsz)] = {k: timeit(f, xb, a.repeats) for k, f in paths.items()}
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
