"""Optimizers.  Adam here is mask-aware; SGD+momentum comes in Step 2."""
from __future__ import annotations

import numpy as np

from engine.parameter import Parameter


class Adam:
    """Adam with PER-ELEMENT step counts and an explicit policy for masked weights.

    Moment policy (the answer to "what happens to m, v on prune / revive"):
      * While an entry is masked:   m = v = t = 0   (state is *dropped*, not frozen).
      * On revival:                 it therefore restarts from m = v = t = 0.
    Why not the alternatives?
      * Keep stale m, v:   the revived weight would first move along a gradient
        direction from before it was pruned (possibly opposite to the current one) -
        momentum from a different loss landscape.  Demonstrated in tests/test_masked.py.
      * Freeze m, v (keep but don't update): same staleness, just delayed.
    Why per-element t: Adam's bias correction 1/(1-beta^t) assumes moments started at 0 at
    step 0.  A revived entry restarts from 0, so with a GLOBAL t its first update would be
    lr*(1-b1)/sqrt(1-b2) = 3.16*lr once t >> 1/(1-b2) (measured: 0.74x at t=56, 2.5x at
    t=1000, 3.16x at t>=1e4) - a spike whose size depends on when you revive.  With its own
    t=1 the first update is ~lr*sign(g): bounded, independent of the global step, and in the
    descent direction of the CURRENT gradient.

    Only `p.grad` (the update gradient) is ever read.  `p.dense_grad` is never touched.
    `reset_state_on_mask_change=False` gives the stale-state ablation used in tests.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, reset_state_on_mask_change=True):
        self.params: list[Parameter] = list(params)
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.reset = reset_state_on_mask_change
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = [np.zeros(p.data.shape, dtype=np.int64) for p in self.params]
        # beta^t per element, maintained by running product (a pow over a big int array each step was
        # the single largest cost in the training step).  float64 so 1-beta2^t is exact for small t.
        self.b1p = [np.ones(p.data.shape) for p in self.params]
        self.b2p = [np.ones(p.data.shape) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def _clear(self, i: int, where) -> None:
        self.m[i][where] = 0.0
        self.v[i][where] = 0.0
        self.t[i][where] = 0
        self.b1p[i][where] = 1.0
        self.b2p[i][where] = 1.0

    def step(self) -> None:
        for i, p in enumerate(self.params):
            m, v, t, b1p, b2p = self.m[i], self.v[i], self.t[i], self.b1p[i], self.b2p[i]
            mask = getattr(p, "mask", None)
            if self.reset:
                ch = p.consume_mask_changes() if hasattr(p, "consume_mask_changes") else None
                if ch is not None:  # prune->revive between two steps still gets a clean slate
                    self._clear(i, ch)
            g = p.grad
            dt = p.data.dtype
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += ((1 - self.b2) * g) * g
            t += 1
            b1p *= self.b1
            b2p *= self.b2
            inv_bc1 = (1.0 / (1.0 - b1p)).astype(dt)  # bias correction in float64, applied in param dtype
            inv_bc2 = (1.0 / (1.0 - b2p)).astype(dt)
            denom = np.sqrt(v * inv_bc2)
            denom += self.eps
            p.data -= (self.lr * m * inv_bc1) / denom
            if mask is not None:
                p.data[~mask] = 0.0  # invariant: pruned weights are exactly 0 after every step
                if self.reset:
                    self._clear(i, ~mask)
