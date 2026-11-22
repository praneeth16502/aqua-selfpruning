# AQUA — The Self-Pruning Network

From-scratch NumPy reverse-mode autodiff + a network that prunes itself during training.

## Restrictions honoured
* No PyTorch / TensorFlow / JAX / Keras / HuggingFace / any autodiff, pruning or NN library.
* **scikit-learn is used ONLY to download/load MNIST** (`fetch_openml`) in `train/data.py`.
  It is never used for models, training, optimisation or gradients.
* Autodiff engine, backward pass, optimizer, masking, pruning and training loop are all in this repo.

## Layout
```
engine/    Tensor, Parameter (masked weights), ops, finite-difference gradcheck
nn/        layers, mask-aware Adam, sparse inference path (compaction + CSR)
prune/     importance criteria (saliency, fisher, magnitude), cubic schedule, mask manager + regrowth
train/     data, training loop, Part 2/3 scripts, sweep, summary, Pareto, cost benchmark
tests/     gradient checks, broadcast, softmax-CE, masked-weight correctness, pruning, sparse path
results/   raw numbers + plots: part2_*, part3/ (every run), part4/ (Pareto, cost, saved models)
```

## Run
```bash
uv venv --python 3.12 && uv pip install -r requirements.txt   # or: pip install -r requirements.txt (Python >= 3.10)
python -m pytest -q                                  # all tests
python -m pytest -q tests/test_gradcheck.py          # Part 1: gradient-check suite
python -m pytest -q tests/test_masked.py             # Part 1: masked-weight correctness
python -m tests.run                                  # stdlib-only fallback runner (no pytest needed)
```

**Part 2: dense training + learning curve** (first run downloads MNIST via sklearn and caches `data/mnist_784.npz`)
```bash
python -m train.train_dense --dataset mnist --epochs 20 --seed 0     # -> results/part2_mnist_seed0.{json,png}
python -m train.init_study                                            # init justification -> results/init_study.json
```

**Part 3: self-pruning run** (Fisher criterion, 90% of weights removed)
```bash
python -m train.train_prune --target 0.9 --criterion fisher --seed 0  # -> results/part3/fisher_s90_seed0.{json,png}
```
Other criteria with the same protocol: `--criterion saliency | magnitude`, `--target 0` for the dense reference,
`--regrow` for regrowth. Protocol (identical for every run): 15 epochs; epochs 0-2 dense warmup, 2-10 cubic sparsity
ramp with masks recomputed every 100 steps, 10-15 fine-tune with masks frozen at exactly the target; final-epoch
model, test set touched once.

**Part 4: Pareto sweep** (fisher vs magnitude, 0/50/75/90/95%, seeds 0-2; ~25 runs, resumable, ~3 min each)
```bash
python -m train.sweep --targets 0 0.5 0.75 0.9 0.95 --criteria fisher magnitude --seeds 0 1 2 --jobs 6 && python -m train.pareto
```
-> `results/part4/pareto.{json,png}`. The 90% comparison over 5 seeds (the falsifiable claim) and its paired tests:
```bash
python -m train.sweep --targets 0 0.9 --criteria saliency fisher magnitude --seeds 0 1 2 3 4 --jobs 6 && python -m train.summarize
```

**Part 4: real cost measurement** (sparse-aware paths vs dense; single-threaded, batch sizes 1 / 32 / 1000)
```bash
python -m train.sweep --targets 0 0.5 0.75 0.9 0.95 --criteria fisher --seeds 0 --out results/part4 --save-model --jobs 5
python -m train.sweep --targets 0.9 --criteria magnitude --seeds 0 --out results/part4 --save-model --jobs 1
python -m train.bench_cost --models results/part4/models/*.npz       # -> results/part4/cost.json
```
Reproducibility: every `python -m train.*` command pins BLAS to one thread (thread count changes float summation order
and therefore the final model; see DESIGN.md appendix C). Numbers reproduce bit-for-bit on the same BLAS build.
Training uses float32; gradient checks run in float64.

## Results (details and derivations in DESIGN.md, working log in notes.md)
| at 90% sparsity, n = 5 seeds | test acc | live weights |
|---|---|---|
| dense (same 15-epoch protocol) | 97.81 ± 0.12% | 234,752 |
| saliency `EMA|w·g|` | 97.85 ± 0.10% | 23,475 |
| fisher `w²·EMA(g²)` (the method) | 97.97 ± 0.13% | 23,475 |
| magnitude `|w|` (baseline) | **98.13 ± 0.05%** | 23,475 |

**Three dense numbers, three different things, not a disagreement.** 97.87% is the Part 2 run: 20 epochs, seed 0 only.
97.81 ± 0.12% is the dense reference under the 15-epoch pruning protocol, seeds 0-4 (the table above). 97.77 ± 0.14% is
the same 15-epoch dense run averaged over seeds 0-2 only, so the Pareto curve uses the same seeds at every point.

**Falsifiable claim.** At 90% sparsity on MNIST (784-256-128-10, this protocol), magnitude pruning retains
98.13 ± 0.05% test accuracy vs 97.85 ± 0.10% for first-order saliency, averaged over 5 seeds; paired difference
−0.28 pp, negative on 5/5 seeds, t = −6.64 (4 df). Fisher vs magnitude: −0.16 pp, t = −2.51 (not significant).
Our importance-based criteria do **not** beat magnitude on accuracy here.

**Cost.** Only structural removal turned into wall-clock speedup. Fisher's 90% model compacts to 434-203-126-10 and runs
1.50× faster than dense at batch 1000 (2.43× at 95%), identical predictions. Magnitude's 90% model compacts to
762-256-127-10 and is not faster (0.74×). Our CSR kernel does 10× fewer multiply-adds at 90% but is slower than dense
BLAS at every batch size: in NumPy, unstructured sparsity is not a speedup.

## Status
- [x] Step 1: autodiff engine, broadcast-correct backward, fused softmax-CE, mask-aware `Parameter` + Adam, tests
- [x] Step 2: MLP (784-256-128-10, ReLU, He/LeCun init), Adam, loop, learning curve. Dense MNIST seed 0, 20 epochs:
      **test acc 97.87%, test loss 0.100** (final-epoch model; `results/part2_mnist_seed0.json`). Val loss bottoms at
      0.085 (epoch 7) and rises to 0.146 by epoch 20 while train loss falls to 0.004: the dense model overfits.
      (An earlier Colab run of the same command gave 97.57%; different BLAS, see notes.md.)
- [x] Step 3: saliency / Fisher / magnitude criteria, cubic ramp, global allocation with per-layer floor, exact hard
      budget, regrowth with grace period (tested; no accuracy effect at 90%)
- [x] Step 4: Pareto sweep 0-95% (3 seeds), 5-seed paired comparison at 90%, sparse-aware cost measurement, falsifiable claim

## Masked weights in one paragraph
`Parameter.masked()` returns `where(mask, w, 0)`. Backward stores **two** gradients: `grad = mask·g`
(the only one the optimizer reads) and `dense_grad = g` (saliency / regrowth signal, never fed to the
optimizer). Adam drops moments of pruned entries and restarts revived ones from zero state with a
per-element step count. Details and measurements: `DESIGN.md`.
