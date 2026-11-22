"""Measures how E[h^2] evolves with depth under different inits (backs the init justification in DESIGN.md).
Run:  python -m train.init_study"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from engine import Tensor, no_grad
from nn.layers import MLP


def second_moments(hidden_gain: float, std_override: float | None = None, depth=8, width=512, n_in=784, seed=0):
    rng = np.random.default_rng(seed)
    net = MLP([n_in] + [width] * depth + [10], rng, dtype=np.float64, hidden_gain=hidden_gain)
    if std_override is not None:  # naive fixed-std init
        for l in net.linears():
            l.W.data[...] = rng.normal(0, std_override, l.W.data.shape)
    x = rng.random((256, n_in))  # MNIST-like: non-negative inputs in [0,1]
    out, h = [], Tensor(x)
    with no_grad():
        for layer in net.layers:
            h = layer(h)
            if layer.__class__.__name__ == "ReLU":
                out.append(float((h.data ** 2).mean()))
    return out


def main():
    cfgs = {
        "he (gain=2, ours)": dict(hidden_gain=2.0),
        "xavier/lecun gain=1 under ReLU": dict(hidden_gain=1.0),
        "naive N(0, 0.01^2)": dict(hidden_gain=2.0, std_override=0.01),
        "naive N(0, 0.1^2)": dict(hidden_gain=2.0, std_override=0.1),
    }
    res = {k: second_moments(**v) for k, v in cfgs.items()}
    print(f"E[h^2] after each of 8 ReLU layers (width 512):")
    for k, v in res.items():
        print(f"  {k:34s} " + " ".join(f"{x:9.2e}" for x in v))
    Path("results").mkdir(exist_ok=True)
    Path("results/init_study.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
