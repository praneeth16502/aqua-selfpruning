"""Dataset loading + splitting + minibatching.

scikit-learn is used HERE ONLY, and ONLY to fetch/load a standard dataset (`fetch_openml`, `load_digits`).
No sklearn model, optimiser, splitter, metric or gradient is used anywhere in this repo.

Split protocol (fixed before any tuning):
  * MNIST's canonical order: first 60,000 = training pool, last 10,000 = TEST.
  * The pool is shuffled with a fixed seed; 5,000 become VALIDATION (model selection, early looks),
    the remaining 55,000 are TRAIN. TEST is only touched for final reported numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Split:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    n_features: int
    n_classes: int
    name: str


def load_mnist(cache_dir: str | Path = "data"):
    """Return (X uint8 [70000,784], y uint8 [70000]). Downloads once via sklearn, then caches as .npz."""
    cache = Path(cache_dir) / "mnist_784.npz"
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["y"]
    from sklearn.datasets import fetch_openml  # loading only

    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
    X, y = X.astype(np.uint8), y.astype(np.uint8)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y)
    return X, y


def load_digits_smoke():
    """sklearn's bundled 8x8 digits (no download). SMOKE/CI ONLY - never a reported result."""
    from sklearn.datasets import load_digits  # loading only

    d = load_digits()
    return d.data.astype(np.float64), d.target.astype(np.int64), 16.0


def make_split(X, y, scale: float, n_test: int, n_val: int, seed: int, dtype=np.float32, name="") -> Split:
    """Last `n_test` samples = test; the rest shuffled (seeded) into val + train. Features scaled to [0,1]."""
    X = np.asarray(X)
    y = np.asarray(y).astype(np.int64)
    n = len(X)
    pool = n - n_test
    perm = np.random.default_rng(seed).permutation(pool)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    sc = lambda a: (a.astype(np.float64) / scale).astype(dtype)
    return Split(
        x_train=sc(X[tr_idx]), y_train=y[tr_idx],
        x_val=sc(X[val_idx]), y_val=y[val_idx],
        x_test=sc(X[pool:]), y_test=y[pool:],
        n_features=X.shape[1], n_classes=int(y.max()) + 1, name=name,
    )


def get_dataset(name: str = "mnist", seed: int = 0, dtype=np.float32, cache_dir="data") -> Split:
    if name == "mnist":
        X, y = load_mnist(cache_dir)
        return make_split(X, y, 255.0, n_test=10_000, n_val=5_000, seed=seed, dtype=dtype, name="mnist")
    if name == "digits":  # smoke only
        X, y, scale = load_digits_smoke()
        # sklearn's digits are ordered by writer: a raw "last N" tail is distribution-shifted (measured: 92.3%
        # test vs 98.0% val, 96.6% once shuffled). Shuffle once with a FIXED seed so the tail is iid.
        p = np.random.default_rng(12345).permutation(len(X))
        X, y = X[p], y[p]
        return make_split(X, y, scale, n_test=297, n_val=300, seed=seed, dtype=dtype, name="digits(smoke)")
    raise ValueError(f"unknown dataset {name!r}")


def iterate_minibatches(x, y, batch_size: int, rng: np.random.Generator | None = None, drop_last: bool = False):
    """Yield (xb, yb). Shuffled each call if `rng` is given (one fresh permutation per epoch)."""
    n = len(x)
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    stop = n - (n % batch_size) if drop_last else n
    for s in range(0, stop, batch_size):
        b = idx[s:s + batch_size]
        yield x[b], y[b]
