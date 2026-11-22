# Engineering Notes — Self-Pruning Network

Running log of decisions, bugs caught, and observations as I build this out.

---

## Step 1 — Autodiff Engine & Masked-Weight Correctness

### Core mental model
The whole point of Part 1 isn't just getting a toy autograd working—it's making sure the graph and optimizer don't silently break when weights get zeroed out (masked) mid-training.

The split that made everything click:
- `grad = mask ⊙ ∂L/∂w_eff` → what the optimizer sees. Strictly 0 for pruned weights.
- `dense_grad = ∂L/∂w_eff` → pre-mask gradient kept separately for saliency and regrowth. Never passed to Adam.

If you mix these up, either Adam keeps nudging dead weights (corrupting momentum) or dead weights look permanently dead to any regrowth logic.

### Bugs / subtleties I caught
- **`np.where(mask, w, 0)` instead of `w * mask`:** In IEEE 754, `nan * 0 = nan` and `inf * 0 = nan`. If a pruned slot ever holds garbage, multiplying by 0 poisons the forward pass. `np.where` avoids this completely.
- **Mask snapshot on forward:** Snapshotting the mask inside `masked()` prevents an in-flight backward pass from seeing a different mask if `set_mask()` is called between forward and backward.
- **Adam revival state:** Dropping state (`m = v = t = 0`) on mask change and tracking a *per-element* step count `t` keeps the first post-revival step bounded at `≈ lr * sign(g)`.
  - Verified against two failure modes: stale moments actually push a revived weight *uphill* (`-5.6e-3` vs `+1.0e-2`), and a global step count `t` with zeroed moments causes a revival spike that grows with training time (measured `0.74×` at `t=56`, `2.5×` at `t=1000`, `3.16×` at `t>=10^4`). Had to fix my own docstring because I initially claimed `~3.2×` everywhere.
- **XOR-invisible round trips:** If a weight is pruned and revived between two optimizer steps, comparing `old_mask != new_mask` misses it. Added change tracking in `Parameter` (`consume_mask_changes`) so Adam still resets state.
- **Mutation-testing my own tests:** Everything passed on the first run (38/38 grad-checks, 62/62 suite, 12/12 masked), which is suspicious. Planted intentional bugs to check test coverage:
  - Removing Adam's post-step re-zero guard didn't fail `d1` because `0 / (0 + eps) = 0` anyway. Added `test_d9` to explicitly corrupt a pruned slot.
  - First broadcast mutation only broke the `(N, D)` side; redid it on the `(D,)` bias side and 11 tests went red.

---

## Step 2 — MLP, Initialization, and Dense MNIST Baseline

### Architecture & Init decisions
- **784 → 256 → 128 → 10 (ReLU):** Needs enough redundancy to survive 90–95% sparsity in Part 3; something tiny like `784-64-10` would collapse.
- **He init** (`Var(w) = 2/n_in`) for hidden ReLU layers, **LeCun init** (`1/n_in`) for the final logits layer since there's no ReLU after it. Verified empirically in `init_study.py` over an 8-layer network—He keeps `E[h²]` flat around `~0.3` while gain=1 decays by ~100×.
- **Watch item for Step 3:** He init preserves variance at step 0, but once 90% of fan-in is pruned, activation variance will shrink. Need to keep an eye on whether training recovers naturally or if rescaling matters.

### Performance & dtype findings
- Switched training to `float32` while keeping `float64` for finite-difference grad-checks (validated float32 vs float64 gradients agree within `1e-4` rel L2).
- Profiling surprise: `float64` wasn't the main slowdown—Adam was taking 6ms out of 10ms/step due to hidden `float64` upcasting and per-step integer array `pow()`. Replacing that with running `β^t` products and native param dtype cut Adam down to 2.5ms.

### Dense MNIST Baseline (Colab run)
- `pytest -q`: **91 passed in 1.64s**.
- `train_dense` (20 epochs, batch 128, lr 1e-3, seed 0):
  - **Final test accuracy: 97.57%** (test loss `0.1212`, final-epoch model, no checkpoint picking).
  - **Val accuracy** peaked at **97.86%** around epoch 10 (val loss `0.0875`), then val loss drifted up to `0.1407` while train acc hit `99.77%`.
- **Takeaway:** Classic dense-MLP overfitting after epoch 10. That's actually a good sign for Step 3—there's plenty of spare capacity in these 235k parameters, so gradual pruning should act as a regularizer and hold close to ~97% even at high sparsity.
- Epoch times bounced between ~5s and ~10s on Colab with no pattern in the loss — that's shared-VM noise, not the code. Not using Colab wall-clock for anything in Part 4; cost numbers get measured locally on one fixed setup.

### Moving off Colab to a local repo
- Pulled the project out of the zip into `~/Projects/aqua-selfpruning` so I have one real git repo instead of notebook cells. The `%%writefile` workflow was fine for getting started but impossible to commit cleanly.
- Mac's system Python is 3.9.6 and `pyproject.toml` says `>=3.10`, so the stock interpreter was out. Used `uv` to pull CPython 3.12 into `.venv` (already gitignored).
- Re-ran the suite locally: **91 passed in 6.92s** — same count as Colab, so nothing in the tests depends on the environment. Slower wall time than Colab but this machine is the one I'll use for everything from here on, which is what matters for consistent timing.
- Rewrote the author on the first 18 commits from the placeholder to my noreply address, kept the original dates. Didn't squash — the brief explicitly asks for real history.
- Committed the baseline JSON + learning curve with the numbers in the commit message, then pushed to a private GitHub repo. Will flip it public / add reviewers at submission.

### Decision going into Step 3
- The dense number to beat (or hold) is **97.57% test, final-epoch**. Every pruned run uses the exact same protocol — same 20 epochs, same split seed, final-epoch model — otherwise the comparison is meaningless.
- If pruned models come out slightly *above* dense at mild sparsity, I'll frame that as "pruning removed the overfitting", not "pruning made the model better". Might also log the dense model at 10 epochs as a second reference so that point is visible rather than argued.

---

## Step 3 — Self-Pruning (in progress)

### Schedule decision (locked before writing code)
- Didn't want to prune a finished, overfitted model — that's basically one-shot pruning with extra steps, which the brief calls the weaker answer. Also considered starting from the epoch-10 best-val checkpoint and dropped it: that's using val to pick a start point, and it breaks the "final-epoch, no checkpoint picking" rule I already committed to.
- Went with **15 epochs for every run**: 2 dense warmup → cubic ramp from epoch 2 to 10 → 5 epochs fine-tuning with masks frozen. Masks recomputed every 100 steps (35 prune events).
- Warmup matters for both criteria: at init `|w·g|` and `|w|` are basically random rankings.
- Same budget for dense, saliency and magnitude. Re-ran dense under this exact protocol so I'm not comparing a 15-epoch pruned run against a 20-epoch dense one.

### Design choices
- **Score = EMA of `|w · dense_grad|` per minibatch, not `|w · EMA(g)|`.** The first-order Taylor term is signed, and near convergence the average gradient goes to ~0, so `|w · mean g|` collapses for every weight. Averaging the absolute per-batch product keeps a signal — it's bounded by `sqrt(w² · E[g²])`, i.e. tied to the Fisher approximation of the second-order term. Wrote a test where signed gradients cancel on purpose to pin this down.
- Uses `dense_grad`, never `grad` — this is exactly what the Step 1 split was for.
- **Global ranking + 5% per-layer floor.** Saliency is in loss units, so comparing across layers is fair. The floor is there because the logits layer is only 0.5% of the weights and a global cut could empty it. Same allocation for magnitude so the baseline differs in just the score.
- No regrowth yet — get the basic version right first.

### Things that bit me
- **Clean-venv crash:** first MNIST run locally died — sklearn's `fetch_openml` silently needs pandas with its default parser. Colab ships pandas, so it never showed up there. A reviewer cloning the repo would have hit this. Switched to `parser="liac-arff"`.
  - Sanity check that the data is the same: local epoch 1 is `0.2904 / 0.9164` vs Colab `0.2905 / 0.9163`. Same data, last-digit BLAS differences.
- **My own test was wrong at first.** I expected saliency to remove all weights on always-zero inputs (like MNIST border pixels — they get exactly zero gradient, so Adam never moves them and they keep their random init size, which magnitude can't tell apart from useful weights). It kept 45 of 96. Turned out 5 of 16 hidden units were dead ReLUs, so their weights also scored exactly 0 and ties got broken by index. Saliency wasn't wrong — dead units are useless too. Rewrote the test with a budget equal to the number of zero-score weights: saliency drops every dead-input weight, magnitude keeps some.

### First MNIST numbers (seed 0, single run — NOT a conclusion yet)
| run | test acc | live weights | layer sparsity (784×256 / 256×128 / 128×10) |
|---|---|---|---|
| dense (15 ep) | 97.32% | 234,752 | 0 / 0 / 0 |
| saliency 90% | 97.97% | 23,475 | 92.8% / 75.6% / 18.0% |
| magnitude 90% | **98.16%** | 23,475 | 92.6% / 75.7% / 55.0% |

- Hit exactly 90.00% both times, frozen through fine-tune. Budget works.
- **Magnitude came out ahead of saliency on this seed.** Not what I expected. But 0.19pp is 19 test images, and the binomial SE on a 10k test set at 98% is already ~0.14pp per run — so this is within noise. Can't say anything until I have multiple seeds.
- Both pruned models beat dense. Dense's final epoch is noisy (val dropped to 96.8% at epoch 15) and it's overfitting; pruning acts as a regularizer. Will frame this as "removes overfitting", not "makes the model better".
- Interesting difference: saliency left the logits layer almost intact (18% pruned), magnitude cut 55% there. Same totals in the first two layers.
- **Activation shrink watch item:** hidden-layer E[h²] on val dropped from 0.74 → 0.46 (saliency) / 0.33 (magnitude) in layer 1, and 7.3 → 4.2 in layer 2. So it does shrink, but only ~40–55% despite removing ~93% of layer-1 fan-in — training compensates. Accuracy is fine, so no rescaling for now.

### Next
- Multiple seeds for dense / saliency / magnitude at 90% before trusting the ranking.
- Then regrowth, then the Pareto sweep in Step 4.

### 5 seeds at 90% — magnitude wins, and it's not noise
Ran seeds 1–4 for all three (12 runs, 3 in parallel, ~2 min each). Same 15-epoch protocol, final-epoch model.

| run | test acc (mean ± std, n=5) | min – max |
|---|---|---|
| dense | 97.61 ± 0.17 | 97.32 – 97.77 |
| saliency 90% | 97.83 ± 0.19 | 97.52 – 97.97 |
| magnitude 90% | **98.17 ± 0.07** | 98.09 – 98.26 |

- Compared them **paired by seed** — same seed means same init and same batch order through warmup, so the pairing cancels out what both runs share. Saliency − magnitude per seed: −0.19, −0.59, −0.47, −0.13, −0.30 pp. Mean **−0.34 pp**, t = −3.91 with 4 df (5% critical value 2.78). Magnitude is better on 5/5 seeds. That's a real difference, not seed noise.
- Seed 0 was actually the *closest* seed. Glad I didn't stop at one run.
- Both pruned models still beat dense (97.61), so the "pruning removes overfitting" point holds across seeds.
- Magnitude is also the most stable (std 0.07 vs 0.19).

What I think is going on (hypotheses, not tested yet):
- Saliency doesn't just generalize worse — it **fits the training set worse** too: final train loss ~0.010 for saliency vs ~0.0055 for magnitude on every seed. So it's removing weights the training loss actually needed. That's the opposite of what a loss-based criterion is supposed to do.
- By the second half of the ramp training loss is ~0.02, so per-batch gradients are small and mostly noise from a few hard examples. A 100-step EMA of `|w·g|` is probably ranking by that noise.
- Magnitude under Adam has long memory for free: a weight gets big because its gradient pointed the same way for thousands of steps. That's basically an integrated movement signal, and my saliency window is only ~100 steps.
- Allocation differs too: saliency keeps the logits layer 73–82% dense, magnitude only ~42–45%. Small in absolute count (~400 weights), but worth noting.

Decision: magnitude stays the **baseline** and saliency stays **the method** — not swapping labels after seeing results. Next I'll try one principled fix (the second-order/Fisher form `w²·E[g²]`, which is what the derivation actually points to late in training) as a measured comparison, and report whatever comes out. If magnitude still wins, that's the honest result and the writeup explains why.

### Second-order run — set up before seeing results
- Added `fisher`: score = `w² · EMA(g²)`, i.e. the `½ H_ii w²` term with the empirical Fisher standing in for the Hessian diagonal. When the average gradient is ~0, `E[g²]` still isn't.
- Same EMA decay (0.99), same 15-epoch protocol, same 90%, same 5 seeds, same split. Only the formula changes. Reusing the magnitude and saliency runs I already have — 5 new runs, not 10.
- Not going to tune the window after seeing it. One shot, reported either way.

### Regrowth rule — written down before running anything
Applies to whichever of saliency / fisher ends up as "the method"; compared against the same method without regrowth, 90%, 5 seeds.
- **Who can come back:** dead weights ranked by EMA of `|dense_grad|`. A revived weight starts at 0, so the first-order loss drop from moving it is proportional to `|g|` — that's the right score for a dead weight (its `|w·g|` is always 0).
- **When / how many:** at each epoch boundary after warmup, revive `k = max(10% of weights pruned during that epoch, 0.5% of live weights)`, and prune the `k` lowest-scoring live weights at the same time — so sparsity stays exactly on schedule (a swap, not extra capacity).
- **Revived at 0 with fresh Adam state** (already how `set_mask` + Adam work).
- **Grace period: revived weights can't be pruned for one epoch.** Caught this while writing the rule — a revived weight is at 0, so its saliency/fisher score is 0 and the very next prune event would kill it again. Without the grace period regrowth would silently do nothing.

### Second-order result (5 seeds, 90%)
| run | test acc (mean ± std, n=5) | min – max | final train loss |
|---|---|---|---|
| dense | 97.61 ± 0.17 | 97.32 – 97.77 | — |
| saliency (1st order) | 97.83 ± 0.19 | 97.52 – 97.97 | ~0.010 |
| fisher (2nd order) | 97.97 ± 0.13 | 97.82 – 98.13 | ~0.0061–0.0076 |
| magnitude | **98.17 ± 0.07** | 98.09 – 98.26 | ~0.0055 |

Paired by seed:
- **fisher − magnitude:** −0.03, −0.29, −0.37, −0.01, −0.30 → mean **−0.20 pp**, t = −2.67 (crit 2.78). Not significant at 5% — but still behind on 5/5 seeds, so I'm not calling it a tie either. "Not significant" with n=5 isn't "equal".
- **fisher − saliency:** +0.16, +0.30, +0.10, +0.12, 0.00 → mean **+0.14 pp**, t = 2.79 (crit 2.78). Borderline significant, never worse on any seed.

What I take from it:
- Second order does what the derivation predicts: it fixes part of the late-training problem. It closes ~40% of the gap to magnitude (−0.34 → −0.20).
- The train-loss clue lines up nicely: fisher's train loss sits between saliency and magnitude, in the same order as test accuracy. So the ranking of criteria is really about how well they keep the weights the training loss needs.
- It doesn't overturn the result. **Magnitude is still the best criterion I have at 90% on MNIST.** The honest falsifiable claim stays the magnitude one.
- Memory-length idea still fits: fisher's EMA window is the same ~100 steps, and magnitude still has the whole run baked in. Didn't test a longer window on purpose — that would be tuning after seeing results.

### Decision: Fisher is "the method" from here on — picked after seeing the 90% results
- Saying this plainly: I chose Fisher over first-order saliency **after** looking at the 90% numbers. It's a choice between two of my own variants tested the same way, and magnitude stays the baseline — but it does mean the Pareto results for Fisher aren't a fully independent confirmation. They're the method I just selected.
- Reasons beyond "it scored higher": it's the term the derivation says should matter near convergence, and it was never worse than saliency on any seed.
- Claims I'll report, both of them, not just the nicer one:
  1. At 90%, magnitude 98.17 ± 0.07% vs saliency 97.83 ± 0.19%, paired −0.34 pp, t = −3.91, n = 5 (significant).
  2. Fisher 97.97 ± 0.13%: vs magnitude −0.20 pp, t = −2.67 (not significant, behind on 5/5); vs saliency +0.14 pp, t = 2.79 (borderline — consistent with the derivation, not a strong win).
- Regrowth gets tested on Fisher only, Fisher+regrowth vs Fisher, 90%, 5 seeds.

### Regrowth result (Fisher + regrowth vs Fisher, 90%, 5 seeds) — no effect
Ran exactly the rule committed beforehand, no changes.

| run | test acc (mean ± std, n=5) | min – max |
|---|---|---|
| fisher | 97.97 ± 0.13 | 97.82 – 98.13 |
| fisher + regrowth | 98.00 ± 0.23 | 97.78 – 98.38 |

- Paired per seed: +0.25, +0.05, −0.11, −0.11, +0.09 → mean **+0.03 pp**, t = 0.50. Nothing there. Regrowth didn't help or hurt; it just made results noisier (std 0.23 vs 0.13).
- The interesting part is *why*: across 13 swaps it revived ~21.8k weights (~18.2k unique) — almost as many as the 23,475 that end up live. Only **~480 of them (~2% of the final network) were still alive at the end**. So ~97% of revived weights got pruned again as soon as their one-epoch grace ran out.
- My read: there's an asymmetry in the rule. A dead weight gets back in on `|g|` (big gradient = "I'd help"), but once it's live it's judged on `w² · E[g²]`, and after one epoch starting from 0 its `w` is still small compared to weights that have been growing for 10 epochs. So it loses. Regrowth with this scoring mostly churns.
- Things that might fix it (NOT doing them now — that would be tuning after the fact): longer grace, revive at a non-zero value, or judge revived weights relative to their own cohort.
- Takeaway for the writeup: regrowth works mechanically (exact budget, real zeros, fresh Adam state, grace period tested), and I can explain the stability side, but with this rule on MNIST at 90% it buys nothing. Counts as an honest ablation, not a win.
- Side note: regrowth runs took ~30% longer (217s vs 168s, both 5-in-parallel) — the extra `|dense_grad|` EMA every step, and scoring continues through fine-tune.

---

## Step 4 — Pareto, cost, reproducibility (in progress)

### 95% crashed — floor bug
- All six 95% runs died at the end of the ramp. The 5% per-layer floor was ceil-rounded: 10,036 + 1,639 + 64 = 11,739 weights required vs a budget of 11,738. Off by one.
- Worse, even with correct rounding, a 5% floor at 95% sparsity forces *every* layer to exactly 5% — the global ranking would quietly turn into uniform per-layer pruning. The crash was actually doing me a favour.
- Fix: floor = `min(5%, (1 − s)/2)`, rounded down. At ≤90% every layer on MNIST is ≥7% dense so the floor never binds — earlier results unaffected (verified below).

### Reproducibility scare — BLAS threads, not code
- After the fix I re-ran Fisher 90% seed 0 to confirm nothing moved: got **98.08%**, committed number was **98.13%**. Panicked a bit.
- Ran the *pre-fix* code on the same seed: also 98.08%. So it wasn't the fix.
- Difference was the environment: numpy here uses **Apple Accelerate**, which listens to `VECLIB_MAXIMUM_THREADS` and ignores `OMP_NUM_THREADS`. My sweep pinned all four thread vars to 1; my quick recheck only set OMP. Different thread count → different float summation order in matmul → after ~6000 steps, a different model.
- With threads pinned: two fresh runs both gave exactly 98.13%, bit for bit. So training is deterministic for a fixed thread count, and current code reproduces the committed number.
- Fix: `train/__init__.py` pins all BLAS thread vars to 1 before numpy loads, so every `python -m train.*` command is reproducible by default.
- My first batch of runs (dense / saliency / magnitude at 90%, seeds 0–4) used other thread settings, so I'm re-running those 15 under the pinned setting. Expect small changes in individual numbers; the conclusions should hold since the shifts (~0.05pp) are well inside seed spread. Will update the tables with whatever comes out.
- Honest limit: bit-for-bit only on the same BLAS build. A reviewer on Linux/OpenBLAS may see last-digit differences.

### Re-run results (pinned threads) — what changed
The 15 re-runs moved individual numbers by up to ~0.4pp (dense seed 0 went 97.32 → 97.64). Updated 90% table, n=5:

| run | test acc (mean ± std) | min – max |
|---|---|---|
| dense | 97.81 ± 0.12 | 97.64 – 97.92 |
| saliency | 97.85 ± 0.10 | 97.68 – 97.94 |
| fisher | 97.97 ± 0.13 | 97.82 – 98.13 |
| magnitude | **98.13 ± 0.05** | 98.08 – 98.21 |

Paired by seed:
- saliency − magnitude: **−0.28 pp, t = −6.64** — main claim got *stronger*, still 5/5 seeds.
- fisher − magnitude: −0.16 pp, t = −2.51 — not significant, behind or tied on 5/5.
- fisher − saliency: +0.12 pp, t = 2.36 — **no longer significant** (was borderline 2.79 before). So I can't claim Fisher beats first-order anymore; it's "never clearly worse, slightly better on average".
- vs dense: magnitude +0.32 (t = 6.28, real), fisher +0.16 (t = 1.65, no), saliency +0.04 (t = 0.71, no).
- So my earlier "pruning beats dense across the board" was partly an artefact of an unlucky dense seed 0 under multithreading. Corrected claim: **at 90% nothing is lost vs dense for any criterion, and magnitude is actually better than dense.** Good lesson in why I re-ran instead of patching one number.

### Pareto (3 seeds per point, same seeds everywhere)
| target | fisher | magnitude |
|---|---|---|
| 0% (dense) | 97.77 ± 0.14 | 97.77 ± 0.14 |
| 50% | 97.79 ± 0.09 | 97.85 ± 0.43 |
| 75% | 97.98 ± 0.02 | 98.14 ± 0.17 |
| 90% | 97.95 ± 0.16 | 98.15 ± 0.07 |
| 95% | 97.83 ± 0.08 | 98.05 ± 0.03 |

- Flat. At 95% both are still at or above dense. This net is far bigger than MNIST needs, so the curve never shows the cliff. Should be upfront that this Pareto curve can't show where accuracy breaks.
- Magnitude ≥ Fisher at every sparsity, by 0.06–0.22pp. Only 3 seeds per point, so individual points aren't significant; the consistent direction is the signal.

### Cost — the part where Fisher actually wins
Benchmark: single thread, MNIST test set, median of 7 timing blocks. Every path gives **identical predictions and test accuracy** to the dense masked model (checked per model).

First run was wrong: the dense baseline called `np.where(mask, W, 0)` every forward, so even the *unpruned* model showed "compact" as 5x faster. The 0% row caught it — dense and compact are the same network there, so anything but 1.00x is a benchmark bug. Fixed, then also found compacted matrices weren't contiguous (column slicing) which made BLAS slower. After both fixes the 0% row is 1.00x at every batch size.

Multiply-adds per image and wall-clock speedup vs dense (batch 1000):

| model | inputs read | compacted shapes | MACs: dense / compact / csr | compact speedup | csr speedup |
|---|---|---|---|---|---|
| dense | 784 | 784-256-128-10 | 234,752 / 234,752 / 234,752 | 1.00x | 0.00x |
| fisher 50% | 559 | 559-256-126-10 | 234,752 / 176,620 / 117,376 | 1.01x | 0.00x |
| fisher 75% | 466 | 466-253-126-10 | 234,752 / 151,036 / 58,688 | 1.24x | 0.01x |
| fisher 90% | 434 | 434-203-126-10 | 234,752 / 114,940 / 23,475 | **1.50x** | 0.03x |
| fisher 95% | 403 | 403-132-123-10 | 234,752 / 70,662 / 11,738 | **2.43x** | 0.07x |
| magnitude 90% | 762 | 762-256-127-10 | 234,752 / 228,854 / 23,452 | **0.74x** | 0.03x |

What this says:
- **Unstructured sparsity in NumPy doesn't buy wall-clock time.** My CSR kernel does 10x fewer multiply-adds at 90% but is 30x *slower* than dense at batch 1000 — the gather materialises an (N × nnz) array and BLAS is just much better at dense matmul. Even at batch 1 it's slower (0.13 ms vs 0.036 ms). The MAC count is a real reduction in arithmetic, but not a speedup on this hardware/library. Not going to dress that up.
- **The real speedup comes from structure, and only Fisher produces it.** Fisher drops 350 of 784 input pixels and ~50 hidden units at 90%, so the compacted dense network is genuinely smaller: 1.50x faster at 90%, 2.43x at 95%, same predictions. Magnitude at the same 90% only frees 22 pixels and 1 unit — compaction gives nothing, and the input gather makes it *slower* (0.74x).
- Why: pixels that are almost always 0 (borders) give near-zero gradients → Fisher scores every weight on them ~0 → whole rows go. Magnitude can't see this; those weights keep their random init size. Same mechanism as the dead-input test in Step 3, now showing up as real cost.
- **So the honest trade-off at 90%: magnitude is 0.16pp more accurate; Fisher is 1.5x faster in actual wall-clock and 2x fewer MACs on a plain dense path.** That's a more interesting result than "my method wins" would have been.

### DESIGN.md + README, and one more reproducibility fix
- Profiled a training step before writing the bottleneck section, instead of guessing: 11.0 ms/step, of which the 9 GEMMs are only 1.6 ms. **Adam is 47% of the step** (5.1 ms in the loop, 2.4 ms on its own). The per-element step counts I added for correct revival mean 7 arrays per parameter — that's memory traffic, not math. Didn't expect my own correctness fix to be the biggest cost. Wrote it up as the first thing I'd optimise (keep per-element `t` only for recently revived weights).
- Re-ran Part 2 locally with pinned threads: **97.87%** test vs the 97.57% I got on Colab. Same command, same seed — different BLAS (Colab vs Accelerate) over 20 epochs. Same overfitting shape (val loss min 0.085 at epoch 7 → 0.146). Replaced the committed Part 2 result with the local one since that's what a reviewer running the README command on this kind of machine gets; Colab number stays here.
- DESIGN.md now answers the four questions directly; the serving answer is tied to what I measured — ship the compacted model, not CSR, and watch for inputs landing on pixels Fisher pruned (it dropped 350 that were ~always 0 in training; if production data uses them, the pruned model is blind there).

### Pre-submission review fixes
- Checked README/DESIGN for stale pre-rerun numbers (−0.34, t=−3.91; +0.14, t=2.79): none left. They only appear in this log, where the correction follows right after.
- README now explains why there are three dense numbers (97.87 Part 2, 20 epochs, seed 0; 97.81 ± 0.12 15-epoch, seeds 0–4; 97.77 ± 0.14 same runs, seeds 0–2 for the Pareto curve). Looked like a disagreement otherwise.
- Regrowth in DESIGN.md had only a sentence; added the per-seed table (fisher vs fisher+regrowth, +0.03 ± 0.15 pp, t = 0.50).
- Measured the piece of Adam I suspected: the per-element bias correction (`β^t` products, `1/(1−β^t)`, cast) is 0.9 ms of Adam's ~2.4 ms standalone — bigger than moment updates and the final update combined. That's my revival-correctness fix costing time. The in-loop 5.1 ms vs 2.4 ms gap is still labelled a hypothesis (memory traffic) — no hardware counters.

### Figures, and two more measurement bugs on the way
- The three per-run Part 3 curves in the repo were **stale** — drawn from the first runs, before the thread pin. The JSON had been replaced, the PNGs hadn't. Wrote `train/figures.py` so every figure is redrawn from the committed JSON; images and numbers can't drift apart again.
- Added two figures that show what the project actually solved: accuracy vs measured speedup for every model (`results/summary.png`), and how many first-layer weights each input pixel kept (`results/part4/input_pixels.png`). The pixel map is the clearest picture of why Fisher is faster: it keeps a digit-shaped blob and drops the border; magnitude keeps almost all 784 pixels because border weights never move from their random init.
- Needed magnitude models at 50/75/95% for the headline chart. Re-ran them and they **didn't match** the committed numbers (98.01 vs 97.95 etc.). Chased it: epoch-1 loss already differed, so not pruning. Ruled out env size, alignment, shell vs in-process thread vars, file vs devnull output. Finally printed the command the sweep builds: **`train.sweep` defaulted to `--threads 2`**. My Fisher model runs had passed `--threads 1` explicitly; these didn't. Worse, the README sweep commands don't pass it either, so a reviewer would have got numbers that don't match the repo. Default is now 1; re-runs match exactly.
- The cost benchmark was noisy: the 0% row (same network both ways, must be 1.00×) read 1.13×, and dense time for identical work drifted 5.3–7.4 ms between models. Switched to round-robin timing (dense and compact back to back each repeat) — then the 0% row read 1.20×, because CSR in the same rotation allocates a ~1 GB temporary that flushes the cache right before dense runs. Timed CSR separately. Now the 0% row is 0.99–1.04× and magnitude 50/75% (nothing removable) is exactly 1.00×.
- Final cost numbers moved a bit: Fisher 1.14 / 1.34 / **1.53** / **2.43×** at 50/75/90/95%; magnitude 1.00 / 1.00 / **0.92** / 0.99×. Biggest correction: magnitude 90% is 0.92×, not the 0.74× I reported before — the old number was inflated by the timing bias. Conclusion unchanged: only Fisher gets faster.
