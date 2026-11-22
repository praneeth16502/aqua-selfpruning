"""Part 3: self-pruning run.

    python -m train.train_prune --target 0.9 --criterion saliency --seed 0

Protocol (identical for dense, saliency and magnitude, so comparisons are like-for-like):
    15 epochs total, final-epoch model, test set touched once.
    epochs [0, warmup)          dense; the criterion only observes (scores at init are noise)
    epochs [warmup, ramp_end)   cubic ramp 0 -> target, masks recomputed every `prune_every` steps
    epochs [ramp_end, epochs)   masks frozen at exactly the target (hard budget), network fine-tunes
--target 0 runs the same loop with no pruning: that is the dense reference for this protocol.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from engine import Tensor, no_grad
from nn.layers import MLP, ReLU
from prune import CRITERIA, PruneManager
from train.data import get_dataset
from train.loop import evaluate, train
from train.plot import plot_prune_curve


def hidden_second_moments(model: MLP, x: np.ndarray) -> list[float]:
    """E[h^2] after every hidden ReLU on `x` (the Step 2 watch item: does pruning fan-in shrink activations?)."""
    out = []
    with no_grad():
        h = Tensor(x)
        for layer in model.layers:
            h = layer(h)
            if isinstance(layer, ReLU):
                out.append(float((h.data.astype(np.float64) ** 2).mean()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.9, help="final sparsity of the weight matrices (0 = dense)")
    ap.add_argument("--criterion", default="saliency", choices=sorted(CRITERIA))
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "digits"])
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--warmup-epochs", type=float, default=2)
    ap.add_argument("--ramp-end-epoch", type=float, default=10)
    ap.add_argument("--prune-every", type=int, default=100, help="steps between mask updates during the ramp")
    ap.add_argument("--ema", type=float, default=0.99, help="EMA decay for saliency / fisher")
    ap.add_argument("--regrow", action="store_true", help="enable regrowth swaps (rule in prune/manager.py)")
    ap.add_argument("--min-density", type=float, default=0.05, help="per-layer floor on kept fraction")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0, help="model init + shuffling")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="data")
    ap.add_argument("--out", default="results/part3")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)

    data = get_dataset(a.dataset, seed=a.split_seed, dtype=np.float32, cache_dir=a.cache_dir)
    dims = [data.n_features, *a.hidden, data.n_classes]
    model = MLP(dims, np.random.default_rng(a.seed), dtype=np.float32)
    spe = math.ceil(len(data.x_train) / a.batch_size)
    start, end = int(a.warmup_epochs * spe), int(a.ramp_end_epoch * spe)
    crit = CRITERIA[a.criterion]() if a.criterion == "magnitude" else CRITERIA[a.criterion](a.ema)
    mgr = PruneManager(model.prunable(), crit, a.target, start, end, a.prune_every, a.min_density,
                       regrow=a.regrow, regrow_every=spe)
    n_params = sum(p.data.size for p in model.parameters())
    print(f"{data.name}: dims {dims} | {n_params:,} params ({mgr.total:,} prunable) | target {a.target:.0%} "
          f"{a.criterion} | prune steps {start}..{end} every {a.prune_every} ({spe} steps/epoch)")

    hist = train(model, data, a.epochs, a.batch_size, a.lr, a.seed, on_backward=mgr.on_backward,
                 on_step=mgr.on_step, epoch_metrics=lambda: {"sparsity": mgr.sparsity()})
    test_loss, test_acc = evaluate(model, data.x_test, data.y_test)  # touched ONCE, final-epoch model
    layer_sp = mgr.layer_sparsity()
    print(f"FINAL test loss {test_loss:.4f} acc {test_acc:.4f} | sparsity {mgr.sparsity():.4f} "
          f"({mgr.active():,}/{mgr.total:,} weights live) | per layer {[round(s, 4) for s in layer_sp]}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = "dense" if a.target == 0 else f"{a.criterion}{'-regrow' if a.regrow else ''}_s{round(a.target * 100)}"
    tag = f"{name}_seed{a.seed}"
    rec = {"config": vars(a), "dims": dims, "steps_per_epoch": spe, "prune_start_step": start, "prune_end_step": end,
           "history": hist, "test_loss": test_loss, "test_acc": test_acc,
           "sparsity": mgr.sparsity(), "active_weights": mgr.active(), "prunable_weights": mgr.total,
           "layer_sparsity": layer_sp, "hidden_E_h2_val": hidden_second_moments(model, data.x_val),
           "events": mgr.events, "regrowth": mgr.regrowth_stats() if mgr.regrow else None,
           "regrow_events": mgr.regrow_events}
    (out / f"{tag}.json").write_text(json.dumps(rec, indent=2))
    if not a.no_plot:
        plot_prune_curve(hist, str(out / f"{tag}_curve.png"), f"{name} {dims} seed {a.seed} | test {test_acc:.2%}",
                         a.warmup_epochs, a.ramp_end_epoch)
    if mgr.regrow:
        print(f"regrowth: {mgr.regrowth_stats()}")
    print(f"wrote {out}/{tag}.json")
    return rec


if __name__ == "__main__":
    main()
