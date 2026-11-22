"""Mini-batch training loop (Adam from nn/optim.py, autodiff from engine/)."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from engine import Tensor, no_grad, softmax_cross_entropy
from nn.optim import Adam
from train.data import Split, iterate_minibatches


def evaluate(model, x, y, batch_size: int = 1000) -> tuple[float, float]:
    """(mean cross-entropy, accuracy). No graph is built."""
    tot_loss, correct = 0.0, 0
    with no_grad():
        for xb, yb in iterate_minibatches(x, y, batch_size):
            logits = model(Tensor(xb))
            tot_loss += float(softmax_cross_entropy(logits, yb).data) * len(yb)
            correct += int((logits.data.argmax(axis=1) == yb).sum())
    return tot_loss / len(x), correct / len(x)


def train(model, data: Split, epochs: int, batch_size: int = 128, lr: float = 1e-3, seed: int = 0,
          log: Callable[[str], None] | None = print, on_backward=None, on_step=None, opt: Adam | None = None):
    """Returns a history dict of per-epoch lists.

    train_loss/train_acc are running means over the epoch's minibatches (weights change during the epoch,
    so they are slightly pessimistic vs. an end-of-epoch pass); val_* are exact end-of-epoch evaluations.
    A non-finite loss raises immediately: a silent NaN is worse than a crash.
    `on_backward(step)` / `on_step(step)` are extension points (Step 3 hooks pruning here).
    """
    rng = np.random.default_rng(seed)
    opt = opt or Adam(model.parameters(), lr=lr)
    hist = {k: [] for k in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "grad_norm", "seconds")}
    step = 0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        sl = sc = sg = 0.0
        n = nb = 0
        for xb, yb in iterate_minibatches(data.x_train, data.y_train, batch_size, rng):
            opt.zero_grad()
            logits = model(Tensor(xb))
            loss = softmax_cross_entropy(logits, yb)
            if not np.isfinite(loss.data):
                raise FloatingPointError(f"non-finite loss {float(loss.data)} at epoch {epoch}, step {step}")
            loss.backward()
            if on_backward:
                on_backward(step)
            gn = float(np.sqrt(sum(float((p.grad.astype(np.float64) ** 2).sum()) for p in model.parameters())))
            opt.step()
            if on_step:
                on_step(step)
            sl += float(loss.data) * len(yb)
            sc += int((logits.data.argmax(axis=1) == yb).sum())
            sg += gn
            n += len(yb)
            nb += 1
            step += 1
        vl, va = evaluate(model, data.x_val, data.y_val)
        if not np.isfinite(vl):
            raise FloatingPointError(f"non-finite validation loss at epoch {epoch}")
        for k, v in zip(hist, (epoch, sl / n, sc / n, vl, va, sg / nb, time.perf_counter() - t0)):
            hist[k].append(v)
        if log:
            log(f"epoch {epoch:3d}  train loss {sl/n:.4f} acc {sc/n:.4f} | val loss {vl:.4f} acc {va:.4f} | "
                f"|g| {sg/nb:.3f} | {hist['seconds'][-1]:.1f}s")
    return hist
