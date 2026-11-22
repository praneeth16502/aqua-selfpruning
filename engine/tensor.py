"""Reverse-mode autodiff on NumPy arrays.

Conventions
-----------
* Every op builds a node with `_parents` and a `_backward(g)` closure that pushes
  `g` (dL/d out) into each parent via `parent._accum(...)`.
* Gradients ACCUMULATE (`+=`).  Never assign.  A tensor used twice must receive
  the sum of both contributions.
* `_accum` is shape-strict: whatever an op passes must already have the parent's
  shape.  Ops that broadcast in the forward pass must `_unbroadcast` in backward.
* `backward()` resets the grads of non-leaf nodes first (they are scratch space),
  but leaf grads accumulate across calls (like PyTorch) until `zero_grad()`.
"""
from __future__ import annotations

import numpy as np


def _unbroadcast(g: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce gradient `g` to `shape`: the adjoint of NumPy broadcasting.

    Forward broadcasting *copies* a value to many positions, so backward must
    *sum* the gradients over every position that copy went to:
      * leading axes that were prepended      -> sum them away
      * axes where the operand had size 1     -> sum, keepdims
    e.g. (N,D) + (D,): grad of the (D,) operand is g.sum(axis=0).
    """
    g = np.asarray(g)
    if g.shape == tuple(shape):
        return g
    extra = g.ndim - len(shape)
    if extra < 0:
        raise ValueError(f"cannot unbroadcast grad of shape {g.shape} to {shape}")
    if extra > 0:
        g = g.sum(axis=tuple(range(extra)))
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and g.shape[i] != 1)
    if axes:
        g = g.sum(axis=axes, keepdims=True)
    return g.reshape(shape)


def _expand_reduced(g, in_shape: tuple, axis, keepdims: bool) -> np.ndarray:
    """Adjoint of a sum-reduction: put reduced axes back and broadcast to in_shape."""
    g = np.asarray(g)
    if axis is not None and not keepdims:
        axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
        axes = sorted(a % len(in_shape) for a in axes)
        for a in axes:
            g = np.expand_dims(g, a)
    return np.broadcast_to(g, in_shape)


_GRAD_ENABLED = True


class no_grad:
    """Context manager: ops inside build no graph (evaluation / inference). Re-entrant and exception-safe."""

    def __enter__(self):
        global _GRAD_ENABLED
        self._prev, _GRAD_ENABLED = _GRAD_ENABLED, False

    def __exit__(self, *exc):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._prev
        return False


def _as_tensor(x) -> "Tensor":
    return x if isinstance(x, Tensor) else Tensor(x)


class Tensor:
    __array_priority__ = 1000  # make `ndarray + Tensor` defer to Tensor.__radd__

    def __init__(self, data, requires_grad: bool = False, dtype=None):
        arr = np.asarray(data, dtype=dtype)
        if dtype is None and not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float64)
        self.data: np.ndarray = arr
        self.grad: np.ndarray | None = None
        self.requires_grad = bool(requires_grad)
        self._parents: tuple = ()
        self._backward = None
        self._op = ""

    # ------------------------------------------------------------------ graph
    @staticmethod
    def _make(data, parents, op: str, backward) -> "Tensor":
        needs = _GRAD_ENABLED and any(p.requires_grad for p in parents)
        out = Tensor(data, requires_grad=needs)
        if needs:  # don't build graph for constants-only subexpressions
            out._parents, out._backward, out._op = tuple(parents), backward, op
        return out

    def _accum(self, g) -> None:
        if not self.requires_grad:
            return
        g = np.asarray(g)
        if g.shape != self.data.shape:
            raise ValueError(
                f"grad shape {g.shape} != tensor shape {self.data.shape} (op={self._op or 'leaf'}); "
                "missing _unbroadcast?"
            )
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += g  # accumulate; never alias/assign

    def zero_grad(self) -> None:
        self.grad = None

    def backward(self, grad=None) -> None:
        if grad is None:
            if self.data.size != 1:
                raise ValueError("backward() on a non-scalar needs an explicit grad")
            grad = np.ones_like(self.data)

        # iterative DFS post-order => topological order, no recursion limit
        topo, seen, stack = [], set(), [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in seen:
                    stack.append((p, False))

        for node in topo:
            if node._parents:  # non-leaf: scratch grads must not leak between calls
                node.grad = None
        self._accum(np.broadcast_to(np.asarray(grad, dtype=self.data.dtype), self.data.shape))
        for node in reversed(topo):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)

    # ------------------------------------------------------------- properties
    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        return self.transpose()

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op or 'leaf'}, requires_grad={self.requires_grad})"

    # -------------------------------------------------------------- arithmetic
    def __add__(self, other):
        other = _as_tensor(other)
        a, b = self, other

        def bw(g):
            a._accum(_unbroadcast(g, a.data.shape))
            b._accum(_unbroadcast(g, b.data.shape))

        return Tensor._make(a.data + b.data, (a, b), "add", bw)

    def __sub__(self, other):
        other = _as_tensor(other)
        a, b = self, other

        def bw(g):
            a._accum(_unbroadcast(g, a.data.shape))
            b._accum(_unbroadcast(-g, b.data.shape))

        return Tensor._make(a.data - b.data, (a, b), "sub", bw)

    def __mul__(self, other):
        other = _as_tensor(other)
        a, b = self, other

        def bw(g):
            a._accum(_unbroadcast(g * b.data, a.data.shape))
            b._accum(_unbroadcast(g * a.data, b.data.shape))

        return Tensor._make(a.data * b.data, (a, b), "mul", bw)

    def __truediv__(self, other):
        other = _as_tensor(other)
        a, b = self, other

        def bw(g):
            a._accum(_unbroadcast(g / b.data, a.data.shape))
            b._accum(_unbroadcast(-g * a.data / (b.data * b.data), b.data.shape))

        return Tensor._make(a.data / b.data, (a, b), "div", bw)

    def __neg__(self):
        a = self

        def bw(g):
            a._accum(-g)

        return Tensor._make(-a.data, (a,), "neg", bw)

    __radd__ = lambda self, o: _as_tensor(o) + self
    __rsub__ = lambda self, o: _as_tensor(o) - self
    __rmul__ = lambda self, o: _as_tensor(o) * self
    __rtruediv__ = lambda self, o: _as_tensor(o) / self

    # ------------------------------------------------------------ linear algebra
    def matmul(self, other):
        other = _as_tensor(other)
        a, b = self, other
        if a.data.ndim != 2 or b.data.ndim != 2:
            raise ValueError(f"matmul supports 2-D @ 2-D only, got {a.data.shape} @ {b.data.shape}")

        def bw(g):
            a._accum(g @ b.data.T)  # (N,K)@(K,D) -> (N,D)
            b._accum(a.data.T @ g)  # (D,N)@(N,K) -> (D,K)

        return Tensor._make(a.data @ b.data, (a, b), "matmul", bw)

    __matmul__ = matmul

    def transpose(self):
        a = self
        if a.data.ndim != 2:
            raise ValueError("transpose supports 2-D only")

        def bw(g):
            a._accum(g.T)

        return Tensor._make(a.data.T, (a,), "transpose", bw)

    # --------------------------------------------------------------- reductions
    def sum(self, axis=None, keepdims: bool = False):
        a = self
        in_shape = a.data.shape

        def bw(g):
            a._accum(_expand_reduced(g, in_shape, axis, keepdims))

        return Tensor._make(a.data.sum(axis=axis, keepdims=keepdims), (a,), "sum", bw)

    def mean(self, axis=None, keepdims: bool = False):
        a = self
        in_shape = a.data.shape
        out = a.data.mean(axis=axis, keepdims=keepdims)
        count = a.data.size // max(np.asarray(out).size, 1)  # number of elements averaged per output

        def bw(g):
            a._accum(_expand_reduced(g, in_shape, axis, keepdims) / count)

        return Tensor._make(out, (a,), "mean", bw)

    # ------------------------------------------------------------ non-linearities
    def relu(self):
        a = self
        pos = a.data > 0  # subgradient at exactly 0 is 0 (documented choice)

        def bw(g):
            a._accum(g * pos)

        return Tensor._make(a.data * pos, (a,), "relu", bw)

    def tanh(self):
        a = self
        t = np.tanh(a.data)

        def bw(g):
            a._accum(g * (1.0 - t * t))

        return Tensor._make(t, (a,), "tanh", bw)

    def sigmoid(self):
        a = self
        x = a.data
        # numerically stable: never exponentiate a large positive number
        s = np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))

        def bw(g):
            a._accum(g * s * (1.0 - s))

        return Tensor._make(s, (a,), "sigmoid", bw)
