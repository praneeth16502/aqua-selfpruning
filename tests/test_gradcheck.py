"""Part 1: every op vs central finite differences (thresholds fixed in engine/gradcheck.py)."""
import numpy as np

from engine import Tensor, gradcheck, softmax_cross_entropy

rng = np.random.default_rng(1234)


def r(*shape):
    return rng.normal(size=shape)


def away_from_zero(*shape, gap=0.1):
    x = rng.normal(size=shape)
    x[np.abs(x) < gap] = 0.5  # keep ReLU away from its kink: finite differences are invalid there
    return x


def positive(*shape):
    return rng.uniform(0.5, 2.0, size=shape)


N, D, K = 5, 4, 3
LABELS = rng.integers(0, K, size=N)

# name -> (fn, input arrays).  Includes the (N,D)+(D,) broadcast case and its siblings.
CASES = {
    "add_same": (lambda a, b: a + b, [r(N, D), r(N, D)]),
    "add_bcast_row_bias(N,D)+(D,)": (lambda a, b: a + b, [r(N, D), r(D)]),
    "add_bcast_(D,)+(N,D)": (lambda a, b: a + b, [r(D), r(N, D)]),
    "add_bcast_(N,D)+(N,1)": (lambda a, b: a + b, [r(N, D), r(N, 1)]),
    "add_bcast_(N,1)+(1,D)": (lambda a, b: a + b, [r(N, 1), r(1, D)]),
    "add_bcast_3d+(2,1)": (lambda a, b: a + b, [r(3, 2, D), r(2, 1)]),
    "add_scalar": (lambda a, b: a + b, [r(N, D), r(1)]),
    "sub_same": (lambda a, b: a - b, [r(N, D), r(N, D)]),
    "sub_bcast(N,D)-(D,)": (lambda a, b: a - b, [r(N, D), r(D)]),
    "mul_same": (lambda a, b: a * b, [r(N, D), r(N, D)]),
    "mul_bcast(N,D)*(D,)": (lambda a, b: a * b, [r(N, D), r(D)]),
    "mul_bcast(N,1)*(1,D)": (lambda a, b: a * b, [r(N, 1), r(1, D)]),
    "div_same": (lambda a, b: a / b, [r(N, D), positive(N, D)]),
    "div_bcast(N,D)/(D,)": (lambda a, b: a / b, [r(N, D), positive(D)]),
    "div_bcast(N,1)/(1,D)": (lambda a, b: a / b, [r(N, 1), positive(1, D)]),
    "neg": (lambda a: -a, [r(N, D)]),
    "matmul": (lambda a, b: a @ b, [r(N, D), r(D, K)]),
    "matmul_square": (lambda a, b: a @ b, [r(D, D), r(D, D)]),
    "transpose": (lambda a: a.T, [r(N, D)]),
    "sum_all": (lambda a: a.sum(), [r(N, D)]),
    "sum_axis0": (lambda a: a.sum(axis=0), [r(N, D)]),
    "sum_axis1": (lambda a: a.sum(axis=1), [r(N, D)]),
    "sum_axis1_keepdims": (lambda a: a.sum(axis=1, keepdims=True), [r(N, D)]),
    "sum_tuple_axes_3d": (lambda a: a.sum(axis=(0, 2)), [r(3, N, D)]),
    "mean_all": (lambda a: a.mean(), [r(N, D)]),
    "mean_axis0": (lambda a: a.mean(axis=0), [r(N, D)]),
    "mean_axis1_keepdims": (lambda a: a.mean(axis=1, keepdims=True), [r(N, D)]),
    "mean_tuple_axes_3d": (lambda a: a.mean(axis=(0, 2)), [r(3, N, D)]),
    "relu": (lambda a: a.relu(), [away_from_zero(N, D)]),
    "tanh": (lambda a: a.tanh(), [r(N, D)]),
    "sigmoid": (lambda a: a.sigmoid(), [r(N, D)]),
    "sigmoid_large_inputs": (lambda a: a.sigmoid(), [rng.normal(size=(N, D)) * 30]),
    "softmax_ce": (lambda z: softmax_cross_entropy(z, LABELS), [r(N, K)]),
    "softmax_ce_large_logits": (lambda z: softmax_cross_entropy(z, LABELS), [r(N, K) * 50]),
    # whole-model compositions: broadcast bias + matmul + nonlinearity + loss, all params checked
    "mlp_relu_ce": (
        lambda x, w1, b1, w2, b2: softmax_cross_entropy((x @ w1 + b1).relu() @ w2 + b2, LABELS),
        [r(N, D), away_from_zero(D, 6) * 0.7, r(6) * 0.0 + 0.3, r(6, K), r(K)],
    ),
    "mlp_tanh_ce": (
        lambda x, w1, b1, w2, b2: softmax_cross_entropy((x @ w1 + b1).tanh() @ w2 + b2, LABELS),
        [r(N, D), r(D, 6), r(6), r(6, K), r(K)],
    ),
    "mse_mean_of_squares": (lambda a, b: ((a - b) * (a - b)).mean(), [r(N, D), r(D)]),
    "layernorm_like_div_mean": (
        lambda a: (a - a.mean(axis=1, keepdims=True)) / ((a * a).mean(axis=1, keepdims=True) + 1.0),
        [r(N, D)],
    ),
}


def _make(name):
    fn, inputs = CASES[name]

    def t():
        gradcheck(fn, inputs, name=name)

    return t


for _name in CASES:  # one named test per op, so a failure says exactly which op broke
    _slug = "".join(c if c.isalnum() else "_" for c in _name)
    globals()[f"test_gradcheck_{_slug}"] = _make(_name)
