"""Part 2: dense baseline.   python -m train.train_dense --dataset mnist --epochs 20 --seed 0"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nn.layers import MLP
from train.data import get_dataset
from train.loop import evaluate, train
from train.plot import plot_learning_curve


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "digits"])
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0, help="model init + shuffling")
    ap.add_argument("--split-seed", type=int, default=0, help="train/val split (kept fixed across model seeds)")
    ap.add_argument("--cache-dir", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)

    data = get_dataset(a.dataset, seed=a.split_seed, dtype=np.float32, cache_dir=a.cache_dir)
    dims = [data.n_features, *a.hidden, data.n_classes]
    model = MLP(dims, np.random.default_rng(a.seed), dtype=np.float32)
    print(f"{data.name}: train {len(data.x_train)} val {len(data.x_val)} test {len(data.x_test)} | dims {dims} | "
          f"{sum(p.data.size for p in model.parameters()):,} params")
    hist = train(model, data, a.epochs, a.batch_size, a.lr, a.seed)
    test_loss, test_acc = evaluate(model, data.x_test, data.y_test)  # touched ONCE, final-epoch model
    print(f"FINAL test loss {test_loss:.4f} acc {test_acc:.4f}  (final-epoch model, no checkpoint picking)")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"part2_{data.name.split('(')[0]}_seed{a.seed}"
    (out / f"{tag}.json").write_text(json.dumps(
        {"config": vars(a), "dims": dims, "history": hist, "test_loss": test_loss, "test_acc": test_acc}, indent=2))
    plot_learning_curve(hist, str(out / f"{tag}_curve.png"), f"{data.name} dense {dims} seed {a.seed}")
    print(f"wrote {out}/{tag}.json and {tag}_curve.png")


if __name__ == "__main__":
    main()
