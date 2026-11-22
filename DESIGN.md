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

## 4. Known limits (deliberate, tested or enforced)
* `matmul` and `transpose` are **2-D only**; other ranks raise `ValueError` (no batched matmul needed for an MLP).
* **ReLU subgradient at exactly 0 is 0.** Finite differences are invalid at the kink, so grad-check inputs are
  kept >= 0.1 away from 0 (`away_from_zero` in `tests/test_gradcheck.py`).
* Grad-check is float64-only (tolerance 1e-6 is meaningless in float32); see Step 2 for how float32 is handled.

## 5. Step 2 decisions: architecture, init, dtype, protocol
**Architecture.** 784-256-128-10, ReLU, 235,146 params of which 234,752 (99.83%) are prunable weights (biases are
394 params and never pruned). At 95% sparsity 11,738 weights remain. Single activation (ReLU) => single init (He).

**Init (variance preservation).** For z = sum_i w_i x_i with zero-mean independent w: Var(z) = n_in * Var(w) * E[x^2].
If x = relu(z_prev) and z_prev is symmetric about 0, E[x^2] = Var(z_prev)/2, so keeping Var(z) constant requires
**Var(w) = 2/n_in (He)**. The logits layer is not followed by a ReLU, so the factor 2 does not apply: **Var(w) = 1/n_in**
(LeCun, gain 1). Biases 0. Measured E[h^2] over 8 ReLU layers of width 512 (`python -m train.init_study`,
`results/init_study.json`): He stays at ~0.3 (max/min < 1.6); gain 1 under ReLU halves every layer (1.5e-1 -> 1.3e-3);
N(0,0.01^2) collapses to 4e-14; N(0,0.1^2) explodes to 4e2. Initial loss on the real architecture is ~ln(10) (tested).
Caveats: fan-in mode preserves the *forward* variance only (backward scales with fan_out/fan_in); Adam's per-element
normalisation makes this largely irrelevant. Variance preservation holds at init only: **after pruning, fan-in
shrinks and activations will shrink with it — watch item for Step 3.**

**dtype decision: train in float32, grad-check in float64 — kept strictly separate.** Measured (1 CPU core,
batch 128, 784-256-128-10): float64 13.1 ms/step, float32 6.3 ms/step (~3 s/epoch incl. evaluation).
Float64 was *not* prohibitively slow; the real bottleneck was Adam (6.0 ms of a 10.0 ms float32 step: a float64
upcast + an int-array `pow` for per-element bias correction). Replacing `beta**t` by running products and keeping the
param dtype cut Adam to 2.5 ms. The 1e-6 finite-difference tolerance is float64-only. float32 is validated
differently: a float32 model and a float64 model with identical (rounded) init must produce gradients within relative
L2 error 1e-4, and no tensor may silently upcast (tests/test_nn.py).

**Protocol.** MNIST canonical order: last 10k = TEST (touched once, at the end, final-epoch model — no checkpoint
picking); the first 60k are shuffled with a *fixed split seed* into 55k train / 5k val. Model seed (init + batch
order) is separate, so Step 4's seed variance measures training randomness, not a different dataset.

**Pitfall found.** sklearn's `digits` (smoke dataset only) is ordered by writer: the raw tail gave 92.3% test vs 98.0%
val; after one fixed-seed shuffle 96.6-98.3% depending on the shuffle (n_test=297 => SE ~1%). Not a leak; a
non-iid split. Smoke runs are never reported as results.

**Not yet established:** the dense MNIST baseline number (needs one run with network access; see README).

## Pending (later steps)
Regrowth trade-off to decide in Step 3: re-entering at 0 is stable (no spike, tested) but slow to regrow — choose
the regrowth score/rate/initial value knowingly and ablate it.
Importance criterion derivation · schedule · regrowth stability · autodiff bottlenecks · serving.
