# DESIGN.md  (living document — Step 1 sections filled; Part 3/4 sections added later)

## 1. What is "the gradient of a masked weight" in this engine?
One sentence: **the optimizer sees `grad = mask ⊙ ∂L/∂w_eff` (the true derivative w.r.t. the stored
weight, exactly 0 where pruned); a separate field `dense_grad = ∂L/∂w_eff` keeps the pre-mask signal
for saliency and regrowth and never reaches the optimizer.**

Forward: `w_eff = where(mask, w, 0)` — a real zero (`where`, not `w*mask`: `nan*0 = nan`, so a corrupted
stored value would otherwise poison the output; tested with 1e30 / inf / nan).

| name | value | consumer | why |
|---|---|---|---|
| `p.grad` | `mask ⊙ g` | optimizer | chain rule: `w_eff` does not depend on a pruned `w`, so the derivative is 0 — not an approximation |
| `p.dense_grad` | `g` | saliency `|w·g|` of live weights; regrowth score `|g|` of dead weights | answers "what would the loss do if this connection existed?" |

Conflation fails in both directions: optimizer on `dense_grad` → pruned weights move (and get clobbered
back, corrupting momentum); regrowth from `grad` → dead weights look dead forever.
The mask is snapshotted at forward time, so `set_mask` between forward and backward cannot change an
in-flight graph (test `d7`).

## 2. Adam state for pruned / revived weights
Policy: **drop** — while masked `m = v = t = 0`; revival restarts from zero state with a **per-element step
count** (`t=1` ⇒ first update ≈ `lr·sign(g)`). Revived weights re-enter at value 0.
* Stale moments (ablation flag `reset_state_on_mask_change=False`): measured in `test_d4` — revived weight
  moves **uphill** (−5.6e-3 vs. the correct +1.0e-2) because pre-prune momentum overrides the current gradient.
* Global `t` with zeroed moments: first update is `3.16·lr` for `t ≥ 1e4` (0.74× at t=56, 2.5× at t=1000) —
  a spike that depends on *when* you revive. Per-element `t` removes the dependence.
* Prune→revive between two optimizer steps is invisible to a before/after mask diff (XOR); `Parameter`
  records every change (`consume_mask_changes`) so state is still reset (`test_d5`).
* Defensive invariant: after each step pruned data is re-zeroed. Mutation testing showed my first suite did
  not exercise this guard (only the stale ablation did) → added `test_d9`.

## 3. How the tests were validated (not just "they pass")
* Gradcheck threshold fixed in advance: `|a−n| ≤ 1e-8 + 1e-6·max(|a|,|n|)`, float64, ε=1e-6.
* `test_gradcheck_sensitivity.py` plants 3 classic bugs (missing batch-sum, overwritten accumulation, wrong
  sign) and requires gradcheck to catch them.
* Implementation mutations (8+ variants incl. `w*mask`, optimizer fed `dense_grad`, no moment reset, no
  unbroadcast on either operand, overwrite-vs-accumulate) each turn ≥1 test red.

## Pending (later steps)
Importance criterion derivation · schedule · regrowth stability · autodiff bottlenecks · serving.
