"""Loader logic on synthetic arrays (no download needed)."""
import numpy as np

from train.data import iterate_minibatches, make_split


def _fake(n=1000, d=6):
    X = np.zeros((n, d))
    X[:, 0] = np.arange(n)           # unique id per sample, to check for leakage
    X[:, 1:] = np.random.default_rng(0).integers(0, 256, size=(n, d - 1))
    y = np.arange(n) % 10
    return X, y


def test_split_sizes_disjoint_and_covering():
    X, y = _fake()
    s = make_split(X, y, 255.0, n_test=100, n_val=50, seed=0)
    ids = [set(a[:, 0].round().astype(int)) for a in (s.x_train * 255, s.x_val * 255, s.x_test * 255)]
    assert [len(i) for i in ids] == [850, 50, 100]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    assert ids[2] == set(range(900, 1000))  # test = the LAST n_test samples, never shuffled into train


def test_split_is_deterministic_per_seed_and_differs_across_seeds():
    X, y = _fake()
    a, b, c = (make_split(X, y, 255.0, 100, 50, seed=s) for s in (0, 0, 1))
    assert np.array_equal(a.x_train, b.x_train) and not np.array_equal(a.x_train, c.x_train)
    assert np.array_equal(a.x_test, c.x_test)  # test set is seed-independent by design


def test_dtype_and_range_and_labels_aligned():
    X, y = _fake()
    X[:, 1:] = 255
    s = make_split(X, y, 255.0, 100, 50, seed=0)
    assert s.x_train.dtype == np.float32 and s.x_train[:, 1:].max() == 1.0 and s.x_train.min() >= 0.0
    ids = (s.x_train[:, 0] * 255).round().astype(int)
    assert np.array_equal(s.y_train, ids % 10)  # label still belongs to its row after shuffling


def test_minibatches_cover_every_sample_exactly_once_and_reshuffle():
    x = np.arange(103)[:, None].astype(np.float32)
    y = np.arange(103)
    rng = np.random.default_rng(0)
    e1 = np.concatenate([yb for _, yb in iterate_minibatches(x, y, 16, rng)])
    e2 = np.concatenate([yb for _, yb in iterate_minibatches(x, y, 16, rng)])
    assert sorted(e1) == list(range(103)) and sorted(e2) == list(range(103))
    assert not np.array_equal(e1, e2)
    assert sum(len(yb) for _, yb in iterate_minibatches(x, y, 16, None, drop_last=True)) == 96
