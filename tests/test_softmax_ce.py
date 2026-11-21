"""Fused softmax + cross-entropy: stability, closed-form gradient, invariances."""
import numpy as np

from engine import Tensor, softmax_cross_entropy

rng = np.random.default_rng(7)


def test_closed_form_gradient_is_softmax_minus_onehot_over_N():
    N, C = 6, 5
    z = rng.normal(size=(N, C))
    y = rng.integers(0, C, size=N)
    t = Tensor(z, requires_grad=True)
    softmax_cross_entropy(t, y).backward()
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    onehot = np.eye(C)[y]
    np.testing.assert_allclose(t.grad, (p - onehot) / N, rtol=1e-12, atol=1e-15)


def test_extreme_logits_no_nan_inf():
    y = np.array([0, 1, 2])
    z = np.array([[1000.0, -1000.0, 0.0], [-1000.0, 1000.0, 5.0], [1e4, 1e4, -1e4]])
    t = Tensor(z, requires_grad=True)
    loss = softmax_cross_entropy(t, y)
    loss.backward()
    assert np.isfinite(loss.data) and np.all(np.isfinite(t.grad))


def test_naive_softmax_actually_breaks_where_ours_does_not():
    # documents WHY the fused form exists
    z = np.array([[1000.0, 0.0, -1000.0]])
    with np.errstate(all="ignore"):
        naive = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    assert not np.all(np.isfinite(naive))
    assert np.isfinite(softmax_cross_entropy(Tensor(z), np.array([1])).data)


def test_shift_invariance():
    z = rng.normal(size=(4, 5))
    y = rng.integers(0, 5, size=4)
    a = softmax_cross_entropy(Tensor(z), y).data
    b = softmax_cross_entropy(Tensor(z + 123.456), y).data
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_uniform_logits_give_log_C_and_row_grads_sum_to_zero():
    N, C = 8, 10
    y = rng.integers(0, C, size=N)
    t = Tensor(np.zeros((N, C)), requires_grad=True)
    loss = softmax_cross_entropy(t, y)
    loss.backward()
    np.testing.assert_allclose(loss.data, np.log(C), rtol=1e-12)
    np.testing.assert_allclose(t.grad.sum(axis=1), 0.0, atol=1e-15)


def test_confident_correct_prediction_has_near_zero_loss_and_grad():
    t = Tensor(np.array([[50.0, 0.0, 0.0]]), requires_grad=True)
    loss = softmax_cross_entropy(t, np.array([0]))
    loss.backward()
    assert loss.data < 1e-20 and np.abs(t.grad).max() < 1e-20


def test_rejects_bad_targets():
    for bad in (np.array([0, 3]), np.array([0.0, 1.0]), np.array([-1, 0])):
        try:
            softmax_cross_entropy(Tensor(np.zeros((2, 3))), bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted bad targets {bad}")
