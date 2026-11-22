# AQUA — The Self-Pruning Network

A small reverse-mode autodiff engine written from scratch in NumPy, and an MLP built on it that prunes its own
connections while it trains.

## What's in here, and what isn't
- No PyTorch, TensorFlow, JAX, Keras, HuggingFace, or any other autodiff, pruning or NN library.
- **scikit-learn is only used to download and load MNIST** (`fetch_openml` in `train/data.py`). Nothing else: no
  models, no training, no gradients.
- The autodiff engine, backward pass, optimizer, masking, pruning and training loop are all my own code in this repo.

## How to read this repo
- **README** (this file): how to run everything, and the headline results.
- **[DESIGN.md](DESIGN.md)**: the four questions from the brief, answered with what I measured.
- **[notes.md](notes.md)**: my working log. I kept it as I went, step by step: what I expected, what actually
  happened, the bugs I hit (including in my own tests and benchmarks), and why I made each decision. If you want to
  know *why* something is the way it is, it's probably in there.
- `results/`: every raw number and plot the docs quote.

```
engine/    Tensor, Parameter (masked weights), ops, finite-difference gradcheck
nn/        layers, mask-aware Adam, sparse inference path (compaction + CSR)
prune/     importance criteria (saliency, fisher, magnitude), cubic schedule, mask manager + regrowth
train/     data, training loop, Part 2/3 scripts, sweep, summary, Pareto, cost benchmark
tests/     gradient checks, broadcast, softmax-CE, masked-weight correctness, pruning, sparse path
results/   part2_* (dense run), part3/ (every pruning run), part4/ (Pareto, cost, saved models)
```

## Setup and tests
```bash
uv venv --python 3.12 && uv pip install -r requirements.txt   # or: pip install -r requirements.txt (Python >= 3.10)
python -m pytest -q                                  # all tests
python -m pytest -q tests/test_gradcheck.py          # Part 1: gradient checks
python -m pytest -q tests/test_masked.py             # Part 1: masked-weight correctness
python -m tests.run                                  # same tests without pytest (stdlib only)
```

## Reproducing each part

**Part 2: dense training and learning curve.** The first run downloads MNIST through sklearn and caches it in
`data/mnist_784.npz`.
```bash
python -m train.train_dense --dataset mnist --epochs 20 --seed 0     # -> results/part2_mnist_seed0.{json,png}
python -m train.init_study                                            # init numbers -> results/init_study.json
```

**Part 3: one self-pruning run** (Fisher criterion, 90% of the weights removed).
```bash
python -m train.train_prune --target 0.9 --criterion fisher --seed 0  # -> results/part3/fisher_s90_seed0.{json,png}
```
Swap in `--criterion saliency` or `--criterion magnitude` to compare, use `--target 0` for the unpruned reference, and
add `--regrow` to turn on regrowth. Every run follows the same 15-epoch schedule: 2 epochs dense, then a cubic ramp
up to the target between epochs 2 and 10 (masks updated every 100 steps), then 5 epochs of fine-tuning with the masks
frozen at exactly the target. I always report the final-epoch model, and the test set is only touched once.

**Part 4: Pareto sweep.** Fisher vs magnitude at 0/50/75/90/95% sparsity, seeds 0-2. That's about 25 runs of ~3
minutes each; the sweep skips anything already in `results/`, so it can be stopped and restarted.
```bash
python -m train.sweep --targets 0 0.5 0.75 0.9 0.95 --criteria fisher magnitude --seeds 0 1 2 --jobs 6 && python -m train.pareto
```
This writes `results/part4/pareto.{json,png}`. The 5-seed comparison at 90% behind the claim below:
```bash
python -m train.sweep --targets 0 0.9 --criteria saliency fisher magnitude --seeds 0 1 2 3 4 --jobs 6 && python -m train.summarize
```

**Part 4: cost measurement.** Trains and saves one model per sparsity, then times the dense, compacted and CSR
inference paths single-threaded at batch sizes 1, 32 and 1000, and checks they all give the same predictions.
```bash
python -m train.sweep --targets 0 0.5 0.75 0.9 0.95 --criteria fisher --seeds 0 --out results/part4 --save-model --jobs 5
python -m train.sweep --targets 0.9 --criteria magnitude --seeds 0 --out results/part4 --save-model --jobs 1
python -m train.bench_cost --models results/part4/models/*.npz       # -> results/part4/cost.json
```

A note on reproducibility: every `python -m train.*` command pins BLAS to one thread. I found out the hard way that
the thread count changes the order floats get summed in, and over a few thousand steps that's enough to end up with
a different model (DESIGN.md, appendix C). With the pin, runs reproduce bit for bit on the same BLAS build. Training
is float32; gradient checks run in float64.

## Results

At 90% sparsity, 5 seeds each:

| | test accuracy | weights kept |
|---|---|---|
| dense (same 15-epoch schedule) | 97.81 ± 0.12% | 234,752 |
| saliency `EMA|w·g|` | 97.85 ± 0.10% | 23,475 |
| fisher `w²·EMA(g²)` (the method) | 97.97 ± 0.13% | 23,475 |
| magnitude `|w|` (the baseline) | **98.13 ± 0.05%** | 23,475 |

You'll see three different dense numbers in this repo, and they don't disagree. 97.87% is the Part 2 run (20 epochs,
seed 0 only). 97.81 ± 0.12% is the dense reference under the 15-epoch pruning schedule over seeds 0-4, as in the table.
97.77 ± 0.14% is the same 15-epoch run over seeds 0-2 only, so the Pareto curve uses the same seeds at every point.

**The falsifiable claim.** At 90% sparsity on MNIST (784-256-128-10, the schedule above), magnitude pruning keeps
98.13 ± 0.05% test accuracy against 97.85 ± 0.10% for first-order saliency, over 5 seeds. Paired by seed, the
difference is −0.28 points, negative on all 5 seeds, t = −6.64 (4 degrees of freedom). Fisher vs magnitude is −0.16
points, t = −2.51, which isn't significant. So on accuracy, my importance-based criteria do **not** beat magnitude
here. DESIGN.md §1.2 goes into why.

**Where Fisher does win is cost.** The only pruning that turned into real wall-clock speedup was the kind that removes
whole rows and columns. Fisher's 90% model shrinks to 434-203-126-10 and runs 1.50× faster than dense at batch 1000
(2.43× at 95%), with identical predictions. Magnitude's 90% model only shrinks to 762-256-127-10 and isn't any faster
(0.74×). My CSR kernel does 10× fewer multiply-adds at 90% but is slower than dense BLAS at every batch size. In
NumPy, unstructured sparsity just isn't a speedup.

## Status
- [x] Step 1: autodiff engine, broadcast-correct backward, fused softmax + cross-entropy, mask-aware `Parameter` and
      Adam, tests.
- [x] Step 2: MLP (784-256-128-10, ReLU, He/LeCun init), Adam, training loop, learning curve. Dense MNIST, seed 0,
      20 epochs: **97.87% test accuracy, test loss 0.100** (final-epoch model; `results/part2_mnist_seed0.json`).
      It overfits: validation loss bottoms at 0.085 around epoch 7 and climbs to 0.146 by epoch 20 while training loss
      drops to 0.004. (My earlier Colab run of the same command gave 97.57%; different BLAS, see notes.md.)
- [x] Step 3: saliency, Fisher and magnitude criteria; cubic ramp; global ranking with a per-layer floor; exact hard
      budget; regrowth with a grace period (it works, but made no difference to accuracy at 90%).
- [x] Step 4: Pareto sweep 0-95% (3 seeds), 5-seed paired comparison at 90%, real cost measurement, falsifiable claim.

## Masked weights in one paragraph
`Parameter.masked()` returns `where(mask, w, 0)`, so a pruned weight is a real zero. On the way back I keep **two**
gradients: `grad = mask · g` is the only one the optimizer ever sees, and `dense_grad = g` is the "what if this
connection existed" signal that saliency and regrowth use. When a weight is pruned, Adam throws away its moments; if
it comes back, it starts from a clean state with its own step count. DESIGN.md §2 has the details and measurements.
