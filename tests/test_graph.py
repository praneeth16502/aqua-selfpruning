"""Topological order + gradient accumulation (tensor used more than once)."""
import numpy as np

from engine import Tensor


def test_tensor_used_twice_accumulates():
    x = Tensor([3.0], requires_grad=True)
    (x * x + x).sum().backward()  # d/dx (x^2 + x) = 2x + 1 = 7
    assert x.grad[0] == 7.0


def test_diamond_graph():
    x = Tensor([2.0], requires_grad=True)
    a, b = x * 2.0, x * 3.0
    (a * b).sum().backward()  # 6x^2 -> 12x = 24
    assert x.grad[0] == 24.0


def test_shared_subexpression_visited_once_in_correct_order():
    # h is consumed by two different downstream nodes; its backward must run AFTER both.
    x = Tensor([1.5], requires_grad=True)
    h = x.tanh()
    y = (h * h + h * 2.0).sum()
    y.backward()
    t = np.tanh(1.5)
    np.testing.assert_allclose(x.grad[0], (2 * t + 2.0) * (1 - t * t), rtol=1e-12)


def test_leaf_grads_accumulate_across_backward_calls_until_zero_grad():
    x = Tensor([2.0], requires_grad=True)
    (x * x).sum().backward()
    (x * x).sum().backward()
    assert x.grad[0] == 8.0
    x.zero_grad()
    (x * x).sum().backward()
    assert x.grad[0] == 4.0


def test_intermediate_grads_do_not_leak_between_backward_calls():
    x = Tensor([2.0], requires_grad=True)
    h = x * 3.0
    y = (h * h).sum()
    y.backward()
    g1 = h.grad.copy()
    x.zero_grad()
    y.backward()  # same graph again
    np.testing.assert_array_equal(h.grad, g1)  # not doubled
    assert x.grad[0] == 36.0  # d/dx 9x^2 = 18x


def test_deep_chain_does_not_hit_recursion_limit():
    x = Tensor([1.0], requires_grad=True)
    y = x
    for _ in range(5000):
        y = y + 1.0
    y.sum().backward()
    assert x.grad[0] == 1.0


def test_grad_is_not_aliased_between_parents():
    a, b = Tensor([1.0, 2.0], requires_grad=True), Tensor([3.0, 4.0], requires_grad=True)
    (a + b).sum().backward()
    a.grad[0] = 99.0
    assert b.grad[0] == 1.0


def test_constants_do_not_build_graph():
    out = Tensor([1.0]) * Tensor([2.0])
    assert out._parents == () and not out.requires_grad


def test_backward_nonscalar_requires_explicit_grad():
    x = Tensor([1.0, 2.0], requires_grad=True)
    try:
        (x * 2.0).backward()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    (x * 2.0).backward(np.array([1.0, 10.0]))
    np.testing.assert_array_equal(x.grad, [2.0, 20.0])


def test_no_grad_builds_no_graph_and_restores_state_even_on_exception():
    from engine import Parameter, no_grad
    p = Parameter(np.ones((2, 2)))
    x = Tensor(np.ones((1, 2)))
    with no_grad():
        out = x @ p.masked()
        assert out._parents == () and not out.requires_grad
        with no_grad():  # re-entrant
            pass
        assert (x @ p.masked())._parents == ()
    assert (x @ p.masked()).requires_grad  # restored
    try:
        with no_grad():
            raise RuntimeError
    except RuntimeError:
        pass
    assert (x @ p.masked()).requires_grad  # restored after exception too
