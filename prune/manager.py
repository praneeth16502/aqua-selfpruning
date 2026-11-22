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
    """See module docstring.  Optional regrowth (off by default; rule fixed in notes.md before any run):

    every `regrow_every` steps after warmup (one epoch), SWAP k connections:
      revive the k dead weights with the largest EMA |dense_grad|  (a revived weight starts at 0, so the first-order
      loss drop from letting it move is proportional to |g|; its |w g| or w^2 E[g^2] would be 0 by construction),
      and prune the k lowest-scoring live weights, so the sparsity schedule is unchanged.
      k = max(regrow_frac * #pruned since the last regrowth, regrow_min * #live).
    Revived weights re-enter at 0 with fresh Adam state (Parameter.set_mask + Adam) and are PROTECTED from pruning
    for `grace` steps (default one epoch): their criterion score is 0 at revival, so without a grace period the next
    prune event would remove them again and regrowth would silently do nothing.
    """

    def __init__(self, params: list[Parameter], criterion, final_sparsity: float, start_step: int, end_step: int,
                 every: int = 100, min_density: float = 0.05, regrow: bool = False, regrow_every: int = 0,
                 regrow_frac: float = 0.10, regrow_min: float = 0.005, grace: int | None = None,
                 regrow_decay: float = 0.99):
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
        self.regrow = bool(regrow) and self.final > 0
        if self.regrow and regrow_every <= 0:
            raise ValueError("regrowth needs regrow_every > 0 (steps per epoch)")
        self.regrow_every, self.regrow_frac, self.regrow_min = int(regrow_every), regrow_frac, regrow_min
        self.grace = int(regrow_every if grace is None else grace)
        self.regrow_decay = regrow_decay
        self.gabs = [np.zeros(p.data.shape) for p in self.params] if self.regrow else None
        self.n_obs = 0
        self.revived_at = [np.full(p.data.shape, -10**9, dtype=np.int64) for p in self.params]
        self.ever_revived = [np.zeros(p.data.shape, dtype=bool) for p in self.params]
        self.pruned_since = 0
        self.regrow_events: list[dict] = []

    # ---------------------------------------------------------------- hooks
    def on_backward(self, step: int) -> None:
        if self.final <= 0:
            return
        if step <= self.end or self.regrow:  # with regrowth, swaps continue in the fine-tune phase
            self.criterion.observe(self.params)
        if self.regrow:
            for e, p in zip(self.gabs, self.params):
                e *= self.regrow_decay
                e += (1.0 - self.regrow_decay) * np.abs(p.dense_grad)
            self.n_obs += 1

    def on_step(self, step: int) -> None:
        if self.final <= 0:
            return
        if step in self.prune_steps:
            self.prune_to(cubic_sparsity(step, self.start, self.end, self.final), step)
        if self.regrow and step > self.start and (step + 1) % self.regrow_every == 0:
            self.regrow_swap(step)

    # ---------------------------------------------------------------- core
    def _protected(self, step: int) -> list[np.ndarray]:
        return [(step - r) < self.grace for r in self.revived_at]

    def _scores(self, step: int) -> list[np.ndarray]:
        sc = self.criterion.scores(self.params)
        if self.regrow:
            sc = [np.where(prot, np.inf, s) for s, prot in zip(sc, self._protected(step))]
        return sc

    def _min_keep(self, sparsity: float) -> list[int]:
        # Floor capped at HALF the remaining density: with floor == 1 - s (e.g. 5% at 95%) every layer would be forced
        # to exactly the same density and the global ranking would silently become uniform per-layer pruning.
        # Rounded DOWN so the floors can never exceed the budget (ceil overshot it by 1 weight at 95%).
        floor = min(self.min_density, 0.5 * (1.0 - sparsity))
        return [min(math.floor(floor * n), n) for n in self.sizes]

    def prune_to(self, sparsity: float, step: int = -1) -> None:
        keep_total = self.total - int(round(sparsity * self.total))
        live = [p.mask for p in self.params]
        before = sum(int(m.sum()) for m in live)
        keep_total = min(keep_total, before)  # monotone: the schedule itself never asks for regrowth
        masks = allocate_keep(self._scores(step), live, keep_total, self._min_keep(sparsity))
        for p, m in zip(self.params, masks):
            p.set_mask(m)
        self.pruned_since += before - self.active()
        self.events.append({"step": step, "target": sparsity, "achieved": self.sparsity(),
                            "layers": self.layer_sparsity()})

    def regrow_swap(self, step: int) -> int:
        """Revive top-k dead weights by EMA|dense_grad|, prune k lowest-scoring unprotected live ones. Returns k."""
        live_n = self.active()
        k = max(int(self.regrow_frac * self.pruned_since), int(self.regrow_min * live_n))
        self.pruned_since = 0
        dead_s = np.concatenate([np.where(p.mask, -np.inf, g).ravel() for p, g in zip(self.params, self.gabs)])
        k = min(k, int(np.isfinite(dead_s).sum()))
        if k <= 0:
            return 0
        top = np.argsort(-dead_s, kind="stable")[:k]
        offsets = np.cumsum([0] + self.sizes)
        revive = [np.zeros(p.data.shape, dtype=bool) for p in self.params]
        for l in range(len(self.params)):
            ix = top[(top >= offsets[l]) & (top < offsets[l + 1])] - offsets[l]
            revive[l].ravel()[ix] = True
            self.revived_at[l].ravel()[ix] = step
            self.ever_revived[l].ravel()[ix] = True
        live_plus = [p.mask | r for p, r in zip(self.params, revive)]
        masks = allocate_keep(self._scores(step), live_plus, live_n, self._min_keep(self.sparsity()))
        for p, m in zip(self.params, masks):
            p.set_mask(m)
        assert self.active() == live_n
        self.regrow_events.append({"step": step, "k": k, "sparsity": self.sparsity()})
        return k

    def regrowth_stats(self) -> dict:
        """How many connections were ever revived, and how many of those are still alive at the end."""
        ever = sum(int(e.sum()) for e in self.ever_revived)
        alive = sum(int((e & p.mask).sum()) for e, p in zip(self.ever_revived, self.params))
        return {"swaps": len(self.regrow_events), "revived_total": sum(e["k"] for e in self.regrow_events),
                "unique_revived": ever, "revived_alive_at_end": alive,
                "alive_frac_of_final_live": alive / max(self.active(), 1)}

    def sparsity(self) -> float:
        return 1.0 - sum(int(p.mask.sum()) for p in self.params) / self.total

    def layer_sparsity(self) -> list[float]:
        return [p.sparsity for p in self.params]

    def active(self) -> int:
        return sum(int(p.mask.sum()) for p in self.params)
