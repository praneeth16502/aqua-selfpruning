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
```
