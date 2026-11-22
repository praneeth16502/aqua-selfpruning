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
