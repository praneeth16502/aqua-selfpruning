"""Importance criteria: how a connection earns the right to survive.

Both criteria expose the same interface so the Part 4 baseline differs from the method in exactly one
place:
    observe(params)  - called after every backward (dense_grad is fresh)
    scores(params)   - list of arrays, higher = more important

Saliency (the method)
---------------------
Removing weight w_i means setting it to 0, i.e. a perturbation dw_i = -w_i.  Taylor-expand the loss:

    dL_i = L(w - w_i e_i) - L(w) ~= -g_i w_i + 1/2 H_ii w_i^2

with g_i = dL/dw_i evaluated with the connection present.  That is exactly `dense_grad` (the pre-mask
gradient), NOT `grad`.  The first-order term is |w_i g_i|.

The catch: the first-order term is SIGNED and, near a minimum of the full-batch loss, E[g_i] -> 0, so
|w_i * E[g_i]| collapses toward 0 for every weight and the ranking becomes noise.  We therefore average
the ABSOLUTE per-minibatch product, EMA_t |w_i g_i^(t)|, not the absolute value of the averaged
product.  E|w g_b| stays informative after convergence: by Jensen it upper-bounds |w E g|, and
E|w g_b| <= sqrt(w^2 E[g_b^2]) = sqrt(w^2 F_ii) where F is the empirical Fisher, the standard
approximation of H_ii in the second-order term.  So the score tracks the first-order term early in
training and something monotone-ish in the (Fisher-approximated) second-order term late in training,
for the price of one elementwise multiply per step.  Derivation in DESIGN.md.

Magnitude (the baseline)
------------------------
|w_i|.  Equivalent to the second-order term under the (wrong) assumption H = c*I: it ignores how much
the loss actually cares about each connection.
"""
from __future__ import annotations

import numpy as np

from engine.parameter import Parameter


class Magnitude:
    name = "magnitude"

    def observe(self, params: list[Parameter]) -> None:
        pass

    def scores(self, params: list[Parameter]) -> list[np.ndarray]:
        return [np.abs(p.data.astype(np.float64)) for p in params]


class Saliency:
    """EMA over steps of |w * dense_grad|.  `decay`=0.99 ~ a 100-step window (~1/4 MNIST epoch at batch 128)."""

    name = "saliency"

    def __init__(self, decay: float = 0.99):
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.decay = decay
        self.ema: list[np.ndarray] | None = None
        self.n = 0

    def observe(self, params: list[Parameter]) -> None:
        if self.ema is None:
            self.ema = [np.zeros(p.data.shape) for p in params]
        for e, p in zip(self.ema, params):
            e *= self.decay
            e += (1.0 - self.decay) * np.abs(p.data * p.dense_grad)
        self.n += 1

    def scores(self, params: list[Parameter]) -> list[np.ndarray]:
        if self.ema is None or self.n == 0:
            raise RuntimeError("saliency has no observations yet; prune only after a warmup")
        bc = 1.0 - self.decay ** self.n  # bias-correct the EMA (matters only for the first ~1/(1-decay) steps)
        return [e / bc for e in self.ema]


CRITERIA = {"saliency": Saliency, "magnitude": Magnitude}
