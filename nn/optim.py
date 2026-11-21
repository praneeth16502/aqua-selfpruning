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
    step 0.  A revived entry restarts from 0, so with a global t its first update would be
    ~lr*(1-b1)/sqrt(1-b2) ~ 3.2*lr (spike).  With its own t=1 the first update is exactly
    ~lr*sign(g): bounded and in the descent direction of the CURRENT gradient.

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

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        for p, m, v, t in zip(self.params, self.m, self.v, self.t):
            mask = getattr(p, "mask", None)
            if self.reset:
                ch = p.consume_mask_changes() if hasattr(p, "consume_mask_changes") else None
                if ch is not None:  # prune->revive between two steps still gets a clean slate
                    m[ch] = 0.0
                    v[ch] = 0.0
                    t[ch] = 0
            g = p.grad
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            t += 1
            m_hat = m / (1 - self.b1 ** t)
            v_hat = v / (1 - self.b2 ** t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            if mask is not None:
                p.data[~mask] = 0.0  # invariant: pruned weights are exactly 0 after every step
                if self.reset:
                    m[~mask] = 0.0
                    v[~mask] = 0.0
                    t[~mask] = 0
