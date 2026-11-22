"""PruneManager: plugs into train.loop.train via its on_backward / on_step hooks.

    on_backward(step): dense_grad is fresh -> criterion.observe()
    on_step(step):     after the optimizer step, at scheduled steps, recompute masks -> Parameter.set_mask()
                       (between steps, so no in-flight graph sees a changing mask; Adam drops the state of
                       every changed entry on its next step, see nn/optim.py)

Allocation: GLOBAL ranking across layers with a per-layer minimum density.
  * Global, because saliency is in loss units and therefore comparable across layers: a weight in layer 1 and
    a weight in layer 3 with the same |w g| are (to first order) equally expensive to remove.  Per-layer
    uniform sparsity would force e.g. the 784x256 input layer (mostly border pixels) and the 128x10 logits
    layer to give up the same fraction.
  * Floor, because a global threshold can disconnect a small layer completely (the logits layer is 0.5% of
    the weights); a network with an empty layer outputs a constant.
  The SAME allocation is used for every criterion, so magnitude vs saliency differ only in the score.

No regrowth here: candidates are the currently live weights, so masks only shrink and the schedule is
monotone.  The target at the end of the ramp is hit EXACTLY (integer count of kept weights), and masks are
frozen afterwards (hard budget).
"""
from __future__ import annotations

import math

import numpy as np

from engine.parameter import Parameter
from prune.schedule import cubic_sparsity


def allocate_keep(scores: list[np.ndarray], live: list[np.ndarray], keep_total: int, min_keep: list[int]) -> list[np.ndarray]:
    """Return new boolean masks keeping exactly `keep_total` live entries, with at least min_keep[l] in layer l.

    Phase 1: each layer keeps its own top-min_keep[l] live entries.
    Phase 2: the remaining budget goes to the globally best of the leftover live entries.
    Ties are broken by flat index (deterministic).
    """
    n_live = sum(int(m.sum()) for m in live)
    if keep_total > n_live:
        raise ValueError(f"cannot keep {keep_total} > {n_live} live entries without regrowth")
    if sum(min_keep) > keep_total:
        raise ValueError("per-layer floors exceed the total budget")
    new = [np.zeros(m.shape, dtype=bool) for m in live]
    rest_s, rest_ix = [], []
    for l, (s, m) in enumerate(zip(scores, live)):
        flat_s = np.where(m, s, -np.inf).ravel()
        order = np.argsort(-flat_s, kind="stable")
        k = min(min_keep[l], int(m.sum()))
        new[l].ravel()[order[:k]] = True
        left = order[k:int(m.sum())]
        rest_s.append(flat_s[left])
        rest_ix.append(np.stack([np.full(len(left), l), left], axis=1))
    budget = keep_total - sum(int(n.sum()) for n in new)
    if budget > 0:
        s_all = np.concatenate(rest_s)
        ix_all = np.concatenate(rest_ix)
        pick = np.argsort(-s_all, kind="stable")[:budget]
        for l, i in ix_all[pick]:
            new[l].ravel()[i] = True
    return new


class PruneManager:
    def __init__(self, params: list[Parameter], criterion, final_sparsity: float, start_step: int, end_step: int,
                 every: int = 100, min_density: float = 0.05):
        self.params = list(params)
        self.criterion = criterion
        self.final = float(final_sparsity)
        self.start, self.end, self.every = int(start_step), int(end_step), int(every)
        self.min_density = float(min_density)
        self.sizes = [p.data.size for p in self.params]
        self.total = sum(self.sizes)
        self.events: list[dict] = []
        # prune at start+every, start+2*every, ..., and always exactly at `end`
        self.prune_steps = set(range(self.start + self.every, self.end, self.every)) | {self.end}

    # ---------------------------------------------------------------- hooks
    def on_backward(self, step: int) -> None:
        if self.final > 0 and step <= self.end:  # scores are only needed until the masks are frozen
            self.criterion.observe(self.params)

    def on_step(self, step: int) -> None:
        if self.final > 0 and step in self.prune_steps:
            self.prune_to(cubic_sparsity(step, self.start, self.end, self.final), step)

    # ---------------------------------------------------------------- core
    def prune_to(self, sparsity: float, step: int = -1) -> None:
        keep_total = self.total - int(round(sparsity * self.total))
        floor = min(self.min_density, 1.0 - sparsity)
        min_keep = [min(math.ceil(floor * n), n) for n in self.sizes]
        live = [p.mask for p in self.params]
        keep_total = min(keep_total, sum(int(m.sum()) for m in live))  # monotone: never asks for regrowth
        masks = allocate_keep(self.criterion.scores(self.params), live, keep_total, min_keep)
        for p, m in zip(self.params, masks):
            p.set_mask(m)
        self.events.append({"step": step, "target": sparsity, "achieved": self.sparsity(),
                            "layers": self.layer_sparsity()})

    def sparsity(self) -> float:
        return 1.0 - sum(int(p.mask.sum()) for p in self.params) / self.total

    def layer_sparsity(self) -> list[float]:
        return [p.sparsity for p in self.params]

    def active(self) -> int:
        return sum(int(p.mask.sum()) for p in self.params)
