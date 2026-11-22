"""Sparse-aware inference path for a pruned MLP (Part 4 cost measurement).  NumPy only, our own kernels.

Two ways to turn a mask into less work, kept separate so each can be measured honestly:

1. Compaction (structured, exact).  A hidden unit whose incoming weights are all pruned outputs relu(b) - a
   constant - and a unit with no outgoing weights contributes nothing.  An input pixel with no live outgoing
   weights is never read.  Removing those rows/columns gives SMALLER DENSE matrices: same BLAS matmul, fewer
   FLOPs, real speedup.  A constant-output unit's effect is folded into the next layer's bias, so the result is
   bit-for-bit the same function (up to float summation order).

2. CSR (unstructured).  Each remaining weight matrix is stored output-major: for output j, the input indices and
   values of its live weights.  Forward = gather the needed input columns, multiply by the values, and segment-sum
   per output (np.add.reduceat).  Work is proportional to the number of live weights - no multiply-by-zero.
   Whether that beats a dense BLAS matmul in wall-clock depends on sparsity and batch size; we measure it rather
   than assume it.

FLOPs counted as 2 per multiply-add (live weights) + 1 per bias add + 1 per ReLU compare are NOT included; we report
multiply-adds explicitly so the counting convention is unambiguous.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CSR:
    """Output-major CSR of a (n_in, n_out) weight matrix W_eff:  y[:, j] = sum_k x[:, indices[k]] * data[k]."""
    indptr: np.ndarray    # (n_out + 1,)
    indices: np.ndarray   # (nnz,)  input index of each live weight
    data: np.ndarray      # (nnz,)
    n_in: int
    n_out: int

    @classmethod
    def from_dense(cls, W: np.ndarray, mask: np.ndarray) -> "CSR":
        Wt, Mt = W.T, mask.T                      # (n_out, n_in): row j = weights into output j
        counts = Mt.sum(axis=1)
        indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        rows, cols = np.nonzero(Mt)               # row-major order => grouped by output j
        return cls(indptr, cols.astype(np.int64), Wt[rows, cols].astype(W.dtype), W.shape[0], W.shape[1])

    @property
    def nnz(self) -> int:
        return int(self.indices.size)

    def matmul(self, x: np.ndarray) -> np.ndarray:
        """x (N, n_in) -> (N, n_out).  Work ~ N * nnz."""
        out = np.zeros((x.shape[0], self.n_out), dtype=np.result_type(x, self.data))
        if self.nnz == 0:
            return out
        prod = x[:, self.indices] * self.data     # (N, nnz) gather + scale
        nonempty = np.flatnonzero(np.diff(self.indptr))
        # reduceat over the start of each non-empty segment; empty outputs stay 0
        out[:, nonempty] = np.add.reduceat(prod, self.indptr[nonempty], axis=1)
        return out


@dataclass
class CompactLayer:
    W: np.ndarray         # dense, already compacted (rows = kept inputs, cols = kept outputs), pruned entries = 0
    mask: np.ndarray
    b: np.ndarray
    csr: CSR


class SparseMLP:
    """Inference-only MLP built from trained (W, mask, b) triples.  relu between layers, none after the last."""

    def __init__(self, Ws, masks, bs, compact: bool = True):
        Ws = [np.where(m, W, 0).astype(W.dtype) for W, m in zip(Ws, masks)]
        masks = [m.astype(bool).copy() for m in masks]
        bs = [b.copy() for b in bs]
        n_layers = len(Ws)
        self.in_keep = np.arange(Ws[0].shape[0])
        self.dense_shapes = [W.shape for W in Ws]
        self.dense_params = sum(W.size + b.size for W, b in zip(Ws, bs))
        if compact:
            # 1) input features never read by layer 0
            used_in = masks[0].any(axis=1)
            self.in_keep = np.flatnonzero(used_in)
            Ws[0], masks[0] = Ws[0][used_in], masks[0][used_in]
            # 2) hidden units: dead-in units output the constant relu(b) -> fold into next bias; dead-out units drop
            for l in range(n_layers - 1):
                has_in = masks[l].any(axis=0)
                has_out = masks[l + 1].any(axis=1)
                const = np.maximum(bs[l], 0) * (~has_in)            # value of units with no incoming weights
                bs[l + 1] = bs[l + 1] + const @ Ws[l + 1]            # their (constant) contribution, folded in
                keep = has_in & has_out
                Ws[l], masks[l], bs[l] = Ws[l][:, keep], masks[l][:, keep], bs[l][keep]
                Ws[l + 1], masks[l + 1] = Ws[l + 1][keep], masks[l + 1][keep]
        self.layers = [CompactLayer(W, m, b, CSR.from_dense(W, m)) for W, m, b in zip(Ws, masks, bs)]

    # ----------------------------------------------------------------- forward paths
    def _run(self, x, mm):
        h = x[:, self.in_keep]
        for i, L in enumerate(self.layers):
            h = mm(h, L) + L.b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0)
        return h

    def forward_compact_dense(self, x):
        """Compacted (smaller) dense matrices, BLAS matmul.  Pruned-but-not-removable entries are still x*0."""
        return self._run(x, lambda h, L: h @ L.W)

    def forward_csr(self, x):
        """Compacted + CSR: only live weights are touched."""
        return self._run(x, lambda h, L: L.csr.matmul(h))

    # ----------------------------------------------------------------- cost accounting
    def cost(self) -> dict:
        live = sum(L.csr.nnz for L in self.layers)
        biases = sum(L.b.size for L in self.layers)
        dense_macs = sum(a * b for a, b in self.dense_shapes)
        compact_macs = sum(L.W.size for L in self.layers)
        return {
            "dense_shapes": [list(s) for s in self.dense_shapes],
            "compact_shapes": [list(L.W.shape) for L in self.layers],
            "inputs_read": int(self.in_keep.size),
            "dense_params": int(self.dense_params),
            "active_params": int(live + biases),
            "macs_per_sample": {"dense": int(dense_macs), "compact_dense": int(compact_macs), "csr": int(live)},
        }
