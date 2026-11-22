import numpy as np

from nn.layers import MLP
from train.data import make_split
from train.loop import evaluate, train


def _blobs(n=600, d=10, k=3, seed=0):
    r = np.random.default_rng(seed)
    y = r.integers(0, k, n)
    centers = r.normal(0, 2.0, (k, d))
    X = centers[y] + r.normal(0, 1.0, (n, d))
    X = (X - X.min()) / (X.max() - X.min())
    return make_split(X, y, 1.0, n_test=100, n_val=100, seed=0)


def test_training_reduces_loss_and_learns_separable_task():
    data = _blobs()
    m = MLP([10, 32, 16, 3], np.random.default_rng(0))
    l0, _ = evaluate(m, data.x_val, data.y_val)
    h = train(m, data, epochs=15, batch_size=32, lr=1e-2, seed=0, log=None)
    assert h["val_loss"][-1] < 0.5 * l0 and h["val_acc"][-1] > 0.9
    assert np.all(np.isfinite(h["train_loss"])) and np.all(np.isfinite(h["grad_norm"]))


def test_training_is_deterministic_for_a_fixed_seed():
    data = _blobs()
    runs = [train(MLP([10, 16, 3], np.random.default_rng(5)), data, 3, 32, 1e-2, seed=5, log=None) for _ in range(2)]
    np.testing.assert_allclose(runs[0]["train_loss"], runs[1]["train_loss"], rtol=1e-6)
    np.testing.assert_allclose(runs[0]["val_acc"], runs[1]["val_acc"])


def test_nan_in_data_raises_instead_of_training_silently():
    data = _blobs()
    data.x_train[0, 0] = np.nan
    try:
        train(MLP([10, 8, 3], np.random.default_rng(0)), data, 1, batch_size=len(data.x_train), log=None)
    except FloatingPointError:
        return
    raise AssertionError("NaN loss was not caught")


def test_hooks_called_once_per_step_after_backward_and_after_step():
    data = _blobs()
    calls = []
    train(MLP([10, 8, 3], np.random.default_rng(0)), data, 1, batch_size=100, log=None,
          on_backward=lambda s: calls.append(("b", s)), on_step=lambda s: calls.append(("s", s)))
    steps = -(-len(data.x_train) // 100)
    assert len(calls) == 2 * steps and calls[:2] == [("b", 0), ("s", 0)]
