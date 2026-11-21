"""Trainable tensor with a binary connectivity mask.

THE MASKED-WEIGHT PARADOX, resolved by naming two different gradients
----------------------------------------------------------------------
The layer computes  w_eff = where(mask, w, 0)   (a REAL zero, never "small").
Let g = dL/dw_eff (the gradient flowing back into the masked op). We store:

    p.dense_grad  = g                      "signal gradient"  (pre-mask)
    p.grad        = mask * g               "update gradient"  (post-mask)

* `grad` is the true derivative of the loss w.r.t. the stored parameter `w`
  (w_eff does not depend on masked entries => derivative exactly 0).  Only this
  one may reach the optimizer.
* `dense_grad` answers a *different* question: "if this connection existed (at
  value w_eff), how would the loss respond?"  It is the quantity needed for the
  Taylor saliency |w * g| of live weights and for regrowth scores |g| of dead ones.
  It must NEVER reach the optimizer.

Conflating them breaks things both ways: feed `dense_grad` to the optimizer and
pruned weights silently move (and the mask must keep clobbering them, corrupting
momentum); use only `grad` and regrowth can never trigger since dead weights look dead.
"""
from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data, mask=None, dtype=None):
        super().__init__(data, requires_grad=True, dtype=dtype)
        self.mask: np.ndarray = np.ones(self.data.shape, dtype=bool)
        self.grad = np.zeros_like(self.data)
        self.dense_grad = np.zeros_like(self.data)
        self._mask_touched: np.ndarray | None = None  # entries whose mask changed since the last optimizer step
        if mask is not None:
            self.set_mask(mask)
            self._mask_touched = None  # construction is not a "change" the optimizer must react to

    # ------------------------------------------------------------------ masking
    def set_mask(self, new_mask, revive_value: float = 0.0) -> None:
        """Install a new mask.  Pruned entries become exactly 0.  Newly revived entries
        start at `revive_value` (default 0: a revived connection re-enters as a no-op and
        grows only if the gradient asks for it).  Records which entries changed so the
        optimizer can reset their state (see nn.optim.Adam)."""
        new = np.asarray(new_mask, dtype=bool)
        if new.shape != self.data.shape:
            raise ValueError(f"mask shape {new.shape} != param shape {self.data.shape}")
        changed = new != self.mask
        revived = new & ~self.mask
        self._mask_touched = changed if self._mask_touched is None else (self._mask_touched | changed)
        self.mask = new.copy()
        self.data[~self.mask] = 0.0
        self.data[revived] = revive_value

    def consume_mask_changes(self) -> np.ndarray | None:
        """Optimizer hook: entries whose connectivity changed since the last step (then cleared)."""
        ch, self._mask_touched = self._mask_touched, None
        return ch

    @property
    def sparsity(self) -> float:
        return 1.0 - float(self.mask.mean())

    # ------------------------------------------------------------------ autodiff
    def masked(self) -> Tensor:
        """Differentiable w_eff = where(mask, w, 0).  Use this in the forward pass."""
        p = self
        mask = self.mask.copy()  # snapshot: later set_mask() must not alter an in-flight graph

        def bw(g):
            p.dense_grad += g                      # signal gradient  dL/dw_eff
            p._accum(np.where(mask, g, 0.0))       # update gradient  mask * dL/dw_eff

        # np.where (not w*mask): a pruned entry is a true 0 even if the stored value is inf/nan
        return Tensor._make(np.where(mask, p.data, 0.0), (p,), "masked", bw)

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)
        self.dense_grad = np.zeros_like(self.data)
