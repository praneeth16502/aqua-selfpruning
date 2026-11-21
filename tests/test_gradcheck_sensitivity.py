"""Mutation tests: a gradient checker that cannot fail proves nothing.
We plant the two classic autodiff bugs and require gradcheck to CATCH them."""
import numpy as np

from engine import Tensor, GradCheckError, gradcheck


def buggy_add_no_unbroadcast(a: Tensor, b: Tensor) -> Tensor:
    """(N,D)+(D,) where the (D,) grad takes only the first row instead of summing over N."""
    def bw(g):
        a._accum(g)
        b._accum(g[0])  # BUG: should be g.sum(axis=0)
    return Tensor._make(a.data + b.data, (a, b), "buggy_add", bw)


def buggy_square_overwrite(a: Tensor) -> Tensor:
    """x*x implemented with assignment instead of accumulation when the input is used twice."""
    def bw(g):
        a.grad = 2 * a.data * g  # BUG: overwrites instead of +=
    return Tensor._make(a.data * a.data, (a,), "buggy_sq", bw)


def _must_fail(fn, inputs):
    try:
        gradcheck(fn, inputs)
    except GradCheckError:
        return
    raise AssertionError("gradcheck failed to detect a planted bug")


def test_gradcheck_catches_missing_batch_sum_in_broadcast():
    rng = np.random.default_rng(0)
    _must_fail(buggy_add_no_unbroadcast, [rng.normal(size=(5, 3)), rng.normal(size=3)])


def test_gradcheck_catches_overwritten_accumulation():
    rng = np.random.default_rng(0)
    _must_fail(lambda a: buggy_square_overwrite(a) + a, [rng.normal(size=(3,))])  # a used twice


def test_gradcheck_catches_wrong_sign():
    def bad_sub(a, b):
        return Tensor._make(a.data - b.data, (a, b), "bad_sub", lambda g: (a._accum(g), b._accum(g)))  # BUG: +g for b
    rng = np.random.default_rng(0)
    _must_fail(bad_sub, [rng.normal(size=(3,)), rng.normal(size=(3,))])
