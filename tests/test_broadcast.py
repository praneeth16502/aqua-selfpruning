"""The classic failure: (N,D) + (D,) forward is trivial, backward must SUM over N."""
import numpy as np

from engine import Tensor, gradcheck

N, D = 7, 4


def test_bias_grad_is_summed_over_batch_and_has_bias_shape():
    x = Tensor(np.random.default_rng(0).normal(size=(N, D)), requires_grad=True)
    b = Tensor(np.zeros(D), requires_grad=True)
    (x + b).sum().backward()
    assert b.grad.shape == (D,), b.grad.shape
    assert np.array_equal(b.grad, np.full(D, float(N)))  # each bias element feeds N outputs
    assert np.array_equal(x.grad, np.ones((N, D)))


def test_bias_grad_with_nontrivial_upstream():
    rng = np.random.default_rng(1)
    x, up = rng.normal(size=(N, D)), rng.normal(size=(N, D))
    b = Tensor(rng.normal(size=D), requires_grad=True)
    ((Tensor(x) + b) * Tensor(up)).sum().backward()
    np.testing.assert_allclose(b.grad, up.sum(axis=0), rtol=1e-12)


def test_gradcheck_exact_case_both_operand_orders():
    rng = np.random.default_rng(2)
    gradcheck(lambda a, b: (a + b) * (a + b), [rng.normal(size=(N, D)), rng.normal(size=D)])
    gradcheck(lambda a, b: (a + b) * (a + b), [rng.normal(size=D), rng.normal(size=(N, D))])


def test_matmul_mean_sum_chain_with_bias():
    rng = np.random.default_rng(3)
    gradcheck(lambda x, w, b: ((x @ w + b).tanh()).mean(), [rng.normal(size=(N, D)), rng.normal(size=(D, 3)), rng.normal(size=3)])
    gradcheck(lambda x, w, b: ((x @ w + b).tanh()).sum(axis=0), [rng.normal(size=(N, D)), rng.normal(size=(D, 3)), rng.normal(size=3)])


def test_mean_divides_by_elements_averaged_not_total():
    x = Tensor(np.ones((N, D)), requires_grad=True)
    x.mean(axis=0).sum().backward()  # each element averaged over N rows
    np.testing.assert_allclose(x.grad, np.full((N, D), 1.0 / N))
