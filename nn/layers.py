"""Composable layers on top of the autodiff engine.  Masking is wired in via Parameter.masked()."""
from __future__ import annotations

import numpy as np

from engine import Parameter, Tensor, no_grad


class Module:
    def parameters(self) -> list[Parameter]:
        return []

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, x):
        return self.forward(x)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Linear(Module):
    """y = x @ W_eff + b,  W_eff = where(mask, W, 0)  (real zeros, see engine/parameter.py).

    Init (variance-preserving, derivation in DESIGN.md):
        gain=2.0  "He":    Var(W) = 2/fan_in   -> for a layer FOLLOWED BY ReLU (ReLU halves E[h^2])
        gain=1.0  "LeCun": Var(W) = 1/fan_in   -> for a layer NOT followed by ReLU (the logits layer)
    Weights are sampled in float64 and then cast, so float32/float64 models with the same seed hold
    identical (rounded) values.  Biases start at 0 and are not prunable (they are O(width), not O(width^2)).
    """

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, gain: float = 2.0, dtype=np.float32):
        std = np.sqrt(gain / n_in)
        self.W = Parameter(rng.normal(0.0, std, size=(n_in, n_out)).astype(dtype))
        self.b = Parameter(np.zeros(n_out, dtype=dtype))
        self.n_in, self.n_out = n_in, n_out

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W.masked() + self.b  # (N,in)@(in,out) + (out,)  <- the broadcast case from Step 1

    def parameters(self):
        return [self.W, self.b]


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]


class MLP(Sequential):
    """Linear+ReLU stack; the last Linear has no activation (it produces logits)."""

    def __init__(self, dims, rng: np.random.Generator, dtype=np.float32, hidden_gain: float = 2.0, out_gain: float = 1.0):
        layers = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            last = i == len(dims) - 2
            layers.append(Linear(a, b, rng, gain=out_gain if last else hidden_gain, dtype=dtype))
            if not last:
                layers.append(ReLU())
        super().__init__(*layers)
        self.dims = list(dims)

    def linears(self) -> list[Linear]:
        return [l for l in self.layers if isinstance(l, Linear)]

    def prunable(self) -> list[Parameter]:
        """The weight matrices (this is where >99% of the parameters live). Biases are excluded."""
        return [l.W for l in self.linears()]

    def logits(self, x: np.ndarray) -> np.ndarray:
        with no_grad():
            return self.forward(Tensor(x)).data
