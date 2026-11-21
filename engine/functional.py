"""Fused ops that must NOT be composed from primitives for numerical reasons."""
from __future__ import annotations

import numpy as np

from .tensor import Tensor


def softmax_cross_entropy(logits: Tensor, targets) -> Tensor:
    """Mean cross-entropy of softmax(logits) vs integer class `targets`, fused.

    Forward (stable, log-sum-exp):
        z' = z - max_j z_j                      (shift: softmax is shift-invariant)
        logp_i = z'_i - log sum_j exp(z'_j)
        L = -(1/N) sum_n logp[n, y_n]
    Backward (derived once, no Jacobian):
        dL/dz[n, i] = (softmax(z)[n, i] - 1[i == y_n]) / N
    Composing softmax -> log -> nll separately would (a) overflow/underflow for
    large |z|, and (b) need the full softmax Jacobian in backward.
    """
    z = logits.data
    y = np.asarray(targets)
    if z.ndim != 2 or y.shape != (z.shape[0],):
        raise ValueError(f"expected logits (N,C) and targets (N,), got {z.shape}, {y.shape}")
    if not np.issubdtype(y.dtype, np.integer) or y.min() < 0 or y.max() >= z.shape[1]:
        raise ValueError("targets must be integer class ids in [0, C)")
    n = z.shape[0]
    shifted = z - z.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -logp[np.arange(n), y].mean()
    probs = np.exp(logp)

    def bw(g):
        grad = probs.copy()
        grad[np.arange(n), y] -= 1.0
        logits._accum(grad * (g / n))

    return Tensor._make(loss, (logits,), "softmax_ce", bw)
