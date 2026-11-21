"""Central finite-difference gradient checking.

Pass criterion (fixed BEFORE running anything, never tuned to make a test pass):
    |analytic - numeric| <= ATOL + RTOL * max(|analytic|, |numeric|)   for every element
with float64, EPS=1e-6.  Truncation error of the central difference is O(EPS^2)~1e-12,
round-off ~1e-16/EPS~1e-10, so a correct gradient clears 1e-6 by orders of magnitude,
while real bugs (missing sum, wrong sign, overwritten accumulation) are O(1) errors.
"""
from __future__ import annotations

import numpy as np

from .tensor import Tensor

EPS = 1e-6
RTOL = 1e-6
ATOL = 1e-8


class GradCheckError(AssertionError):
    pass


def numerical_grad(f, x: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Central difference of scalar function f at array x (x is restored on exit)."""
    x = np.array(x, dtype=np.float64)
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    for _ in it:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fp = f(x)
        x[idx] = old - eps
        fm = f(x)
        x[idx] = old
        g[idx] = (fp - fm) / (2 * eps)
    return g


def gradcheck(fn, inputs, eps=EPS, rtol=RTOL, atol=ATOL, seed=0, name="") -> float:
    """Check d(scalarised fn)/d(inputs) analytic vs numeric.

    fn(*tensors) -> Tensor of any shape. Non-scalar outputs are projected onto a fixed
    random tensor R (L = sum(out * R)) so the upstream gradient is non-trivial.
    Returns the worst abs error. Raises GradCheckError on failure.
    """
    arrays = [np.array(a, dtype=np.float64) for a in inputs]
    probe = fn(*[Tensor(a) for a in arrays])
    proj = None
    if probe.data.size > 1:
        proj = np.random.default_rng(seed).normal(size=probe.data.shape)

    def scalar(t: Tensor) -> Tensor:
        return t.sum() if proj is None else (t * Tensor(proj)).sum()

    tensors = [Tensor(a, requires_grad=True) for a in arrays]
    scalar(fn(*tensors)).backward()

    worst = 0.0
    for i, (t, a) in enumerate(zip(tensors, arrays)):
        analytic = t.grad if t.grad is not None else np.zeros_like(a)

        def f(x, i=i):
            args = [Tensor(arr) for arr in arrays]
            args[i] = Tensor(x)
            return float(scalar(fn(*args)).data)

        numeric = numerical_grad(f, a, eps)
        err = np.abs(analytic - numeric)
        tol = atol + rtol * np.maximum(np.abs(analytic), np.abs(numeric))
        worst = max(worst, float(err.max()) if err.size else 0.0)
        if np.any(err > tol):
            k = np.unravel_index(np.argmax(err - tol), err.shape)
            raise GradCheckError(
                f"gradcheck FAILED {name} input#{i} at {k}: analytic={analytic[k]:.8g} "
                f"numeric={numeric[k]:.8g} abs_err={err[k]:.3g} tol={tol[k]:.3g}"
            )
    return worst
