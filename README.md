# AQUA — The Self-Pruning Network

From-scratch NumPy reverse-mode autodiff + a network that prunes itself during training.

## Restrictions honoured
* No PyTorch / TensorFlow / JAX / Keras / HuggingFace / any autodiff, pruning or NN library.
* **scikit-learn is used ONLY to download/load MNIST** (`fetch_openml`) in `train/data.py`.
  It is never used for models, training, optimisation or gradients.
* Autodiff engine, backward pass, optimizer, masking, pruning and training loop are all in this repo.

## Layout
```
engine/   Tensor, Parameter (masked weights), ops, finite-difference gradcheck
nn/       layers + optimizers (Adam)                      
prune/    importance criteria, schedules, mask manager    (Step 3)
train/    data loading + training loops                   (Step 2+)
tests/    gradient checks, broadcast, softmax-CE, masked-weight correctness
```

## Run
```bash
uv venv && uv pip install -r requirements.txt      # or: pip install -r requirements.txt
python -m pytest -q                                  # everything
python -m pytest -q tests/test_gradcheck.py          # gradient-check suite (Part 1)
python -m pytest -q tests/test_masked.py             # masked-weight correctness (Part 1)
python -m tests.run                                  # stdlib-only fallback runner (no pytest needed)

python -m train.train_dense --dataset mnist --epochs 20 --seed 0   # Part 2: dense baseline + learning curve
                                                                    # (first run downloads MNIST via sklearn, caches data/mnist_784.npz)
python -m train.init_study                                          # init justification numbers -> results/init_study.json
```
Training uses float32; gradient checks run in float64 (see DESIGN.md section 5).

## Status
- [x] Step 1: autodiff engine, broadcast-correct backward, fused softmax-CE, mask-aware `Parameter` + Adam, tests
- [x] Step 2: MLP (784-256-128-10, ReLU/He), Adam, loop, curve — dense MNIST seed 0, 20 epochs:
      **test acc 97.57%, test loss 0.121** (final-epoch model, no checkpoint picking; `results/part2_mnist_seed0.json`).
      Val acc peaks at 97.86% (epoch 10); val loss bottoms at 0.0875 (epoch 10) and rises to 0.141 by epoch 20
      while train loss falls to 0.007 — the dense model overfits after ~epoch 10.
- [ ] Step 3: saliency pruning + cubic ramp + regrowth
- [ ] Step 4: Pareto sweep, sparse-aware cost measurement, seeds, falsifiable claim

## Masked weights in one paragraph
`Parameter.masked()` returns `where(mask, w, 0)`. Backward stores **two** gradients: `grad = mask·g`
(the only one the optimizer reads) and `dense_grad = g` (saliency / regrowth signal, never fed to the
optimizer). Adam drops moments of pruned entries and restarts revived ones from zero state with a
per-element step count. Details and measurements: `DESIGN.md`.
