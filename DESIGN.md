# DESIGN.md

The four questions from the brief, answered with what was actually measured in this repo. Raw numbers live in
`results/`; the running log of how I got there (including the mistakes) is `notes.md`.

**Results in one paragraph.** MNIST, 784-256-128-10 ReLU MLP, 15-epoch protocol, final-epoch model, test set touched
once. At 90% sparsity (n = 5 seeds): magnitude 98.13 ± 0.05%, Fisher 97.97 ± 0.13%, first-order saliency
97.85 ± 0.10%, dense 97.81 ± 0.12%. Magnitude beats first-order saliency on every seed (paired −0.28 pp, t = −6.64).
Fisher sits in between and is not significantly different from either. Accuracy is flat up to 95% for both Fisher
and magnitude. On cost, only Fisher's pruned model actually runs faster (1.50× at 90%, 2.43× at 95%, batch 1000,
identical predictions), because it removes whole input pixels and hidden units; magnitude's does not (0.74×).

---

## 1. The importance criterion, and why it approximates "loss increase if removed"

### 1.1 Derivation
Removing connection *i* means setting `w_i → 0`, a perturbation `δw = −w_i e_i`. Taylor-expand the loss around the
current weights:

```
ΔL_i = L(w − w_i e_i) − L(w)
     = −g_i w_i  +  ½ w_i² H_ii  +  O(w_i³)          g_i = ∂L/∂w_i,  H_ii = ∂²L/∂w_i²
```

`g_i` must be the gradient *with the connection present*: for a live weight that is the ordinary gradient, and the
engine stores exactly this quantity as `dense_grad` (§2). The criterion keeps the connections with the largest
estimated |ΔL| and removes the rest.

**First order (`saliency`).** Keep only the first term: `|ΔL_i| ≈ |w_i g_i|`. One elementwise multiply per step.

**Why first order fails late in training.** The first-order term is *signed* and depends on the full-batch
gradient. Near a minimum, `E[g_i] → 0` for every weight, so `|w_i · E[g_i]| → 0` everywhere and the ranking becomes
noise. Two consequences shaped the implementation:
* I average the **absolute per-minibatch** product, `EMA_t |w_i g_i^(t)|`, not `|w_i · EMA_t g_i^(t)|`. By Jensen it
  upper-bounds the latter, and it stays nonzero after convergence because minibatch gradients don't vanish even when
  their mean does. (Test: two minibatches whose gradients cancel for one weight; its score must stay > 0.)
* But that quantity is now measuring gradient *noise*, not the first-order loss change. It is bounded by
  `sqrt(w_i² E[g_i²])`, which points at the second-order term.

**Second order (`fisher`).** Keep the curvature term. The diagonal Hessian is too expensive to compute exactly in
NumPy (one extra backward pass per weight, or a Hessian-vector product per prune event), so use the standard
empirical-Fisher approximation `H_ii ≈ F_ii = E[g_i²]`:

```
|ΔL_i| ≈ ½ w_i² E[g_i²]            score = w_i² · EMA_t (g_i^(t))²     (the ½ doesn't change the ranking)
```

Near convergence the first-order term vanishes but `E[g²]` stays positive, so this term doesn't. Same EMA decay
(0.99, ≈100 steps ≈ 0.23 epoch) as `saliency`, so the comparison between them changes only the formula.

**Magnitude (baseline).** `|w_i|`. This is the second-order term under the assumption `H = c·I`: every weight is
assumed equally important per unit size. It ignores the data entirely.

### 1.2 What the measurements say about the derivation
| criterion (90%, n=5) | test acc | final train loss |
|---|---|---|
| saliency `EMA|w·g|` | 97.85 ± 0.10 | ~0.010 |
| fisher `w²·EMA(g²)` | 97.97 ± 0.13 | ~0.006–0.008 |
| magnitude `|w|` | **98.13 ± 0.05** | ~0.0055 |

* **The derivation-based criteria lose to magnitude on MNIST.** Saliency − magnitude: −0.28 pp, negative on 5/5
  seeds, t = −6.64 (4 df, 5% critical value 2.78). Fisher − magnitude: −0.16 pp, t = −2.51, not significant but
  behind or tied on 5/5. Fisher − saliency: +0.12 pp, t = 2.36, not significant.
* **It is a fitting problem, not a generalisation problem.** The criteria rank the same way on *training* loss as on
  test accuracy. Saliency removes weights the training loss actually needed. Second order recovers part of it, in
  the direction the derivation predicts.
* **Memory length is the most plausible explanation.** Under Adam a weight's magnitude is the sum of thousands of
  updates, so `|w|` integrates the whole training history: a weight is big because its gradient pointed the same way
  for a long time. The EMA criteria remember ~100 steps. Late in the ramp, when training loss is ~0.02 and per-batch
  gradients are mostly noise from a few hard examples, that short memory hurts. Magnitude remembers the whole run;
  saliency only remembers the last quarter epoch. (Not tested: a longer window. Changing it after seeing the result
  would have been tuning on the outcome.)
* **Where the data-aware criteria do win: structure.** Pixels that are almost always 0 give near-zero gradients, so
  every weight on them scores ≈0 under saliency/Fisher, while magnitude keeps them at their random init size. At 90%
  Fisher drops 350 of 784 inputs entirely; magnitude drops 22. That is what makes Fisher's model cheaper to run (§3,
  Part 4). Unit test: on always-zero inputs saliency removes every such weight, magnitude keeps some.
* **Disclosure.** Fisher became "the method" after the 90% comparison (it is the term the derivation points to
  late in training, and was never clearly worse than first order). Magnitude stayed the baseline throughout; I did
  not swap labels. Fisher's Pareto results are therefore not an independent confirmation.

### 1.3 Schedule and allocation
* **Cubic ramp** `s(t) = s_f (1 − (1 − u)³)`, `u` = progress through the ramp. Its slope is largest at the start, when
  the network is most redundant, and zero at the end, when every remaining weight matters and the network needs time
  to re-route. **Protocol, identical for every run:** 15 epochs; epochs 0–2 dense warmup (scores at init are noise
  for every criterion); epochs 2–10 ramp, masks recomputed every 100 steps (35 prune events); epochs 10–15
  fine-tune with masks frozen at exactly the target. One-shot pruning at the end was rejected because it gives the
  network no time to adapt; pruning from the epoch-10 best-validation checkpoint was rejected because it uses
  validation to pick the starting point and breaks the "final-epoch, no checkpoint picking" rule.
* **Global ranking with a per-layer floor.** Saliency and Fisher are in loss units, so they are comparable across
  layers. The floor (`min(5%, (1 − s)/2)`, rounded down) stops a global cut from emptying the 1,280-weight logits
  layer. The cap at half the remaining density matters: at 95% a flat 5% floor would have forced every layer to
  exactly 5%, silently turning global ranking into uniform per-layer pruning (found when 95% crashed on an
  off-by-one in the floor rounding). The same allocation is used for every criterion.
* **Hard budget.** The kept count is an integer, hit exactly at the end of the ramp and held through fine-tune
  (tested). Achieved sparsity is reported from the masks, not the target.

### 1.4 Regrowth and its stability
Rule fixed in `notes.md` before any run. Every epoch after warmup, **swap** `k = max(10% of weights pruned that
epoch, 0.5% of live weights)`: revive the dead weights with the largest `EMA|dense_grad|` and prune the same number of
lowest-scoring live ones, so sparsity stays on schedule.
* **Score for dead weights is `|g|`.** A revived weight enters at 0, so its `|w·g|` or `w²E[g²]` is 0 by
  construction. The first-order loss decrease from letting it move is proportional to `|g|`.
* **Revived at 0, fresh Adam state, per-element step count** (§2.2): the first update is ≈ `lr·sign(g)`, bounded and
  in the current descent direction, with no spike that depends on when the weight was revived.
* **Grace period, one epoch.** Without it, a revived weight scores 0 at the next prune event and is removed straight
  away; regrowth would silently do nothing. Tested both ways.
* **Result: no effect.** Same 90% target, same seeds, same protocol; only `--regrow` differs
  (`results/part3/fisher{,-regrow}_s90_seed*.json`, paired test in `results/part3/summary.json`):

  | seed | 0 | 1 | 2 | 3 | 4 | mean ± std |
  |---|---|---|---|---|---|---|
  | fisher | 98.13 | 97.82 | 97.89 | 98.08 | 97.92 | 97.97 ± 0.13 |
  | fisher + regrowth | 98.38 | 97.87 | 97.78 | 97.97 | 98.01 | 98.00 ± 0.23 |
  | difference (pp) | +0.25 | +0.05 | −0.11 | −0.11 | +0.09 | **+0.03 ± 0.15, t = 0.50** |

  Not significant (critical value 2.78), positive on 3 of 5 seeds, and noisier. Mechanism, from the run logs: about
  21.8k revivals over 13 swaps, but only ~480 revived weights (2% of the final network) survive to the end. The rule is asymmetric: weights get back in on `|g|` and are then judged on `w²E[g²]` against weights that have
  been growing for 10 epochs. With this rule it mostly churns. Candidate fixes (not run, to avoid tuning on the
  outcome): longer grace, non-zero revival value, judging revived weights within their own cohort.

---

## 2. What is "the gradient of a masked weight"?
One sentence: **the optimizer sees `grad = mask ⊙ ∂L/∂w_eff` (the true derivative w.r.t. the stored weight,
exactly 0 where pruned); a separate field `dense_grad = ∂L/∂w_eff` keeps the pre-mask signal for saliency and
regrowth and never reaches the optimizer.**

Forward: `w_eff = where(mask, w, 0)`, a real zero. `where`, not `w*mask`, because `nan*0 = nan`: a corrupted stored
value would otherwise poison the output (tested with 1e30 / inf / nan in pruned slots).

| name | value | consumer | why |
|---|---|---|---|
| `p.grad` | `mask ⊙ g` | optimizer | chain rule: `w_eff` does not depend on a pruned `w`, so the derivative is 0. Not an approximation. |
| `p.dense_grad` | `g` | saliency `|w·g|`, Fisher `w²g²` of live weights; regrowth score `|g|` of dead weights | answers "what would the loss do if this connection existed?" |

Why this is the right choice: the two answer different questions, and conflating them fails both ways. Optimizer on
`dense_grad` → pruned weights move and have to be clobbered back, which corrupts momentum. Regrowth from `grad` →
dead weights look dead forever. The mask is snapshotted at forward time, so `set_mask` between forward and backward
cannot change an in-flight graph (test `d7`).

### 2.1 Adam state for pruned / revived weights
Policy: **drop.** While masked, `m = v = t = 0`. Revival restarts from zero state with a **per-element step count**
(`t = 1` ⇒ first update ≈ `lr·sign(g)`). Revived weights re-enter at value 0.
* Stale moments (ablation flag `reset_state_on_mask_change=False`): measured in `test_d4`, the revived weight moves
  **uphill** (−5.6e-3 vs. the correct +1.0e-2), because pre-prune momentum overrides the current gradient.
* Global `t` with zeroed moments: the first update is `3.16·lr` for `t ≥ 1e4` (0.74× at t=56, 2.5× at t=1000), a
  spike that depends on *when* you revive. A per-element `t` removes the dependence.
* Prune→revive between two optimizer steps is invisible to a before/after mask diff (XOR). `Parameter` records every
  change (`consume_mask_changes`), so the state is still reset (`test_d5`).
* Defensive invariant: after each step pruned data is re-zeroed. Mutation testing showed the first suite did not
  exercise this guard → added `test_d9`.

### 2.2 Whiteboard derivations this rests on
* `(N,D) + (D,)`: the bias is broadcast over N rows, so `∂L/∂b = Σ_n G[n,:]`. `_unbroadcast` sums prepended axes
  and size-1 axes.
* `Y = A B`: `∂L/∂A = G Bᵀ`, `∂L/∂B = Aᵀ G`.
* Softmax + CE fused: `∂L/∂z = (softmax(z) − onehot(y)) / N`, forward via log-sum-exp (stable for logits of 1000;
  naive softmax overflows, tested).
* Masked weight: `w_eff = m·w` elementwise (as a function), so `∂L/∂w = m ⊙ ∂L/∂w_eff`.

---

## 3. Where does the autodiff engine bottleneck, and how would I optimise it?

### 3.1 Training step (measured, 1 thread, batch 128, 784-256-128-10, Fisher scoring on)
| part | ms/step | share |
|---|---|---|
| forward | 1.99 | 18% |
| backward | 2.96 | 27% |
| Adam | 5.12 | 47% |
| Fisher scoring (`observe`) | 0.74 | 7% |
| zero_grad | 0.17 | 2% |
| **total** | **10.98** | |
| the 9 GEMMs this step needs, nothing else | 1.62 | |

The matrix multiplies are ~15% of the step. The rest is overhead:
* **Adam is the single biggest cost.** Measured: 5.1 ms per step inside the training loop, 2.4 ms when the same
  `step()` runs on its own; the mask re-zero/state-clear bookkeeping is 0.09 ms (negligible).
  *Known, measured cost:* the per-element bias correction that makes revival correct (§2.1). Every step, for every
  parameter, it updates two full-size float64 arrays `β1^t, β2^t` (running products; the original `β**t` power over an
  int array was already removed in Step 2), computes `1/(1−β1^t)` and `1/(1−β2^t)` as full arrays and casts them to
  float32. Timed alone that is **0.9 ms of Adam's ~2.4 ms**, more than the moment updates (0.36 ms) and the final
  update (0.25 ms) together. With a global `t` it would be two scalars.
  *Hypothesis, not measured:* the extra 2.7 ms inside the loop is memory traffic. Per element Adam reads/writes
  `data, grad, m, v, t (int64), β1^t, β2^t (float64)` plus temporaries, ~24 bytes more than textbook Adam, and inside
  the loop those arrays compete for cache with activations and gradients. I haven't confirmed this with hardware
  counters.
* **Python-level graph overhead.** Every op allocates a `Tensor`, a closure and output arrays; `backward` walks a
  Python topological order. Fine for an MLP with ~10 nodes, dominant for deep or wide graphs of small ops.
* **NumPy temporaries.** Each elementwise expression allocates a new array (`m*b1 + (1-b1)*g` is three). Backward
  allocates for every gradient, then accumulates with `+=`.
* **Masking costs.** `where(mask, w, 0)` copies every weight matrix on every forward; `dense_grad` is a second
  full-size gradient accumulated every step even though it's only needed while scoring.

Optimisations, in the order I'd do them:
1. **Make bias correction cheap.** Keep one global `t` (two scalars) for weights that were never revived, and a sparse
   side-table of per-element `t` only for recently revived entries; after ~1/(1−β2) ≈ 1000 steps a revived entry's
   correction is ≈1 and it can rejoin the global path. Same revival guarantee, and the 0.9 ms full-array work drops
   to roughly the size of the revived set. Then fuse the remaining elementwise Adam ops into one pass per parameter
   (a compiled kernel; in NumPy, at least `out=` buffers so no temporaries).
2. **Preallocate and update in place.** Gradient buffers, Adam temporaries and the masked-weight buffer reused across
   steps (`np.multiply(..., out=)`), so no per-step allocation.
3. **Fuse `linear + bias + relu`** forward and backward into one op: one node instead of three, one temporary
   instead of three.
4. **Only compute `dense_grad` when it's needed** (on scoring steps, and not at all in fine-tune without
   regrowth); otherwise backward writes `grad` only.
5. Cache `w_eff` between mask changes instead of recomputing `where` every forward.
6. **Inference in NumPy: run the compacted dense model, not CSR.** Measured in §3.2: compaction is 1.50× faster than
   dense at 90% (Fisher), CSR is ~30× slower despite 10× fewer multiply-adds. CSR only pays with a compiled sparse
   kernel, or with structured (block / 2:4) sparsity the hardware supports.

### 3.2 Inference: sparse is not automatically fast (measured, Part 4)
Three paths on each saved model, single thread, MNIST test set, identical predictions checked per model
(`results/part4/cost.json`):
* **dense**: `x @ where(mask, W, 0)`, the dense-times-zeros shortcut. Same work as the unpruned model; **not a speedup.**
* **compact**: drop inputs that no live weight reads, and hidden units with no inputs (their constant `relu(b)` is
  folded into the next bias) or no outputs. The result is smaller dense matrices on the same BLAS.
* **csr**: compact + my own CSR kernel (gather the live inputs per output, multiply, segment-sum). Work is
  proportional to live weights only.

| model (batch 1000) | MACs/image: dense / compact / csr | compact vs dense | csr vs dense |
|---|---|---|---|
| fisher 90% | 234,752 / 114,940 / 23,475 | **1.50×** | 0.03× |
| fisher 95% | 234,752 / 70,662 / 11,738 | **2.43×** | 0.07× |
| magnitude 90% | 234,752 / 228,854 / 23,452 | 0.74× | 0.03× |

* **Unstructured sparsity buys no wall-clock time in NumPy.** CSR does 10× fewer multiply-adds at 90% and is ~30×
  slower than dense at batch 1000 (and still slower at batch 1). The gather materialises an `(N × nnz)` array, and
  dense BLAS is far better at dense matmul than NumPy is at gather + segment-sum. The MAC reduction is real; the
  speedup isn't, on this stack.
* **Structured removal is what pays.** Fisher's 90% model is 434-203-126-10 after compaction: 1.50× faster, same
  predictions. Magnitude's is 762-256-127-10: no smaller, and the input gather makes it slower.
* Two benchmark bugs were caught by the 0%-sparsity row, where dense and compact are the same network and must time
  at 1.00×: the dense path re-masked weights every call (made dense look 5× slower), and compacted matrices were
  non-contiguous after column slicing. Both fixed.

---

## 4. Serving a self-pruned model in a multi-tenant inference service at thousands of RPS

**What to ship, based on what I measured.** In this stack only *structural* pruning turned into real speedup, so the
production artifact is the **compacted dense model** (smaller matrices, plain GEMM), not the CSR one. Unstructured
CSR only becomes worth it with a real sparse kernel (compiled, SIMD, ideally hardware-supported 2:4 or block
sparsity) and at sparsities well above 90%, and I'd measure that before believing it. If I were training for
deployment I'd change the objective to match: prune *blocks or neurons* (score aggregated per row/column/block)
so the accuracy-vs-cost trade-off is measured on the thing that actually gets faster.

**Artifact and versioning.**
* Export immutable, versioned artifacts: compacted weights, kept input/unit indices, folded biases, plus metadata
  (training commit, seed, sparsity, per-layer shapes, eval accuracy, BLAS build). Content-hash them so a rollback is
  a pointer change.
* Every release is gated on the same evaluation as here: accuracy on a held-out set vs the dense parent and vs the
  previous version, plus measured p50/p99 latency on the target hardware. A pruned model that isn't faster in
  measured latency doesn't ship.
* Keep the dense parent around as a fallback for every pruned version.

**Multi-tenant layout.**
* Tenants usually share a base and differ in fine-tuning. Pruning changes the *shape* (compacted dims differ per
  model), so models with different structure can't share one batched GEMM. Options, in order of preference: share
  one structure across tenants and vary only the weights (prune once, fine-tune per tenant under a fixed mask), so
  requests from many tenants batch into the same kernel; or bucket tenants by structure.
* Hot models stay resident; cold ones load on demand with an LRU of decoded artifacts. Compacted models are small
  (Fisher 90% is ~24k weights here), so thousands fit in memory. The cost that matters is the per-request work, not
  storage.
* Isolation: per-tenant concurrency limits and queues so one tenant's burst can't take another's latency budget;
  separate pools for very large or very noisy tenants.

**Throughput and latency.**
* Dynamic batching with a small max-wait (a few ms) per structure bucket. At batch 1000 the compacted Fisher 90%
  model ran in 3.6 ms single-threaded vs 5.5 ms dense, so the speedup survives batching. At batch 1 it is 0.026 vs
  0.036 ms, where request overhead dominates anyway.
* Pin BLAS threads per worker and scale with processes. Same lesson as reproducibility here: oversubscribed BLAS
  threads change both latency and numerics.
* Latency SLOs per tenant (p99), with load shedding and fallback to a cheaper or cached path when the queue grows.

**Monitoring.**
* Per tenant, per model version: latency, throughput, error rate, input distribution stats, and prediction
  distribution vs the dense parent on a shadow-traffic sample.
* Accuracy drift after pruning is the specific risk. The pruned model kept accuracy on the training distribution,
  but it threw away inputs it thought were useless: Fisher dropped 350 pixels that were almost always 0 *in
  training*. If production inputs start using those regions (different crops, devices, shifted data), the pruned
  model is blind there and the dense one isn't. So: monitor the share of input mass landing on pruned inputs, and
  periodically run the dense parent on a sample to measure disagreement. Alert, then fall back to dense, then
  re-prune on fresh data.

---

## Appendix A — Step 2 decisions: architecture, init, dtype, protocol
**Architecture.** 784-256-128-10, ReLU, 235,146 params of which 234,752 (99.83%) are prunable weights (biases are
394 params and never pruned). At 95% sparsity 11,738 weights remain. Single activation (ReLU) ⇒ single init (He).

**Init (variance preservation).** For z = Σ_i w_i x_i with zero-mean independent w: Var(z) = n_in · Var(w) · E[x²].
If x = relu(z_prev) and z_prev is symmetric about 0, E[x²] = Var(z_prev)/2, so keeping Var(z) constant requires
**Var(w) = 2/n_in (He)**. The logits layer is not followed by a ReLU, so the factor 2 does not apply: **Var(w) = 1/n_in**
(LeCun, gain 1). Biases 0. Measured E[h²] over 8 ReLU layers of width 512 (`python -m train.init_study`,
`results/init_study.json`): He stays at ~0.3 (max/min < 1.6); gain 1 under ReLU halves every layer (1.5e-1 → 1.3e-3);
N(0,0.01²) collapses to 4e-14; N(0,0.1²) explodes to 4e2. Initial loss on the real architecture is ~ln(10) (tested).
Caveat: fan-in mode preserves the *forward* variance only; Adam's per-element normalisation makes the backward
mismatch largely irrelevant.
**After pruning** (the Step 2 watch item), measured on val at 90%, seed 0: hidden E[h²] drops from 0.78 → 0.33–0.45
(layer 1) and 8.3 → 4.3 (layer 2). So it shrinks, by roughly 40–60%, despite removing ~93% of layer-1 fan-in:
training compensates. Accuracy is unaffected, so no rescaling.

**dtype: train in float32, grad-check in float64, kept strictly separate.** Measured (1 CPU core, batch 128):
float64 13.1 ms/step, float32 6.3 ms/step. The 1e-6 finite-difference tolerance is float64-only. float32 is validated
differently: a float32 model and a float64 model with identical (rounded) init must produce gradients within relative
L2 error 1e-4, and no tensor may silently upcast (tests/test_nn.py).

**Protocol.** MNIST canonical order: last 10k = TEST (touched once, at the end, final-epoch model, no checkpoint
picking); the first 60k are shuffled with a *fixed split seed* into 55k train / 5k val. The model seed (init + batch
order) is separate, so seed variance measures training randomness, not a different dataset. Same seed ⇒ same init
and same batch order, which is why method comparisons are **paired by seed**.

## Appendix B — Test validation and known limits
* Gradcheck threshold fixed in advance: `|a−n| ≤ 1e-8 + 1e-6·max(|a|,|n|)`, float64, ε=1e-6.
* `test_gradcheck_sensitivity.py` plants 3 classic bugs (missing batch-sum, overwritten accumulation, wrong sign) and
  requires gradcheck to catch them. Implementation mutations (`w*mask`, optimizer fed `dense_grad`, no moment reset,
  no unbroadcast on either operand, overwrite-vs-accumulate, removed bias folding in compaction) each turn ≥1 test red.
* `matmul` and `transpose` are **2-D only**; other ranks raise. ReLU subgradient at exactly 0 is 0, so grad-check
  inputs stay ≥0.1 from the kink.

## Appendix C — Reproducibility
* Every `python -m train.*` entry point pins BLAS to 1 thread before numpy loads (`train/__init__.py`). Measured why:
  Fisher 90% seed 0 gives 98.13% single-threaded and 98.08% with Accelerate's default threading. Different thread
  counts change float summation order, and over ~6,000 steps that lands on a different model. With the pin, reruns
  are bit-identical (every saved model in `results/part4/models` reproduces its `results/part3` number exactly).
* Bit-for-bit holds on the same BLAS build. On a different BLAS (OpenBLAS / MKL) expect last-digit differences,
  well inside the seed spread.
* `fetch_openml` is called with `parser="liac-arff"`: the default needs pandas, which Colab has and a clean
  `requirements.txt` install doesn't.
