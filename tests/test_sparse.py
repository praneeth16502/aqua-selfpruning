"""Part 4: the sparse inference path computes the same function as the masked dense model, with less work."""
import numpy as np

from engine import Tensor, no_grad
from nn.layers import MLP
from nn.sparse import CSR, SparseMLP


def test_csr_matmul_matches_masked_dense_including_empty_rows_and_cols():
    r = np.random.default_rng(0)
    W = r.normal(size=(30, 12))
    mask = r.random((30, 12)) < 0.2
    mask[:, 3] = False          # an output with no inputs
    mask[5, :] = False          # an input that is never read
    W[~mask] = np.nan           # pruned slots hold garbage: must never be read
    x = r.normal(size=(7, 30))
    csr = CSR.from_dense(W, mask)
    want = x @ np.where(mask, W, 0)
    np.testing.assert_allclose(csr.matmul(x), want, rtol=1e-12, atol=1e-12)
    assert csr.nnz == mask.sum()
    assert np.all(csr.matmul(x)[:, 3] == 0)


def test_csr_all_pruned_gives_zeros():
    csr = CSR.from_dense(np.ones((4, 3)), np.zeros((4, 3), bool))
    assert csr.nnz == 0 and np.all(csr.matmul(np.ones((2, 4))) == 0)


def _pruned_mlp(seed=0, density=0.1):
    r = np.random.default_rng(seed)
    m = MLP([50, 40, 30, 5], r, dtype=np.float64)
    for l in m.linears():
        mask = r.random(l.W.data.shape) < density
        l.W.set_mask(mask)
        l.b.data[...] = r.normal(size=l.b.data.shape)   # nonzero biases so folding matters
    lin = m.linears()
    lin[0].W.set_mask(lin[0].W.mask & ~np.eye(50, 40, dtype=bool))
    m0 = lin[0].W.mask.copy(); m0[:, 7] = False; lin[0].W.set_mask(m0)    # hidden unit 7: no inputs -> constant
    lin[0].b.data[7] = 1.5                                                  # ... a nonzero constant, so folding is exercised
    m1 = lin[1].W.mask.copy(); m1[11, :] = False; lin[1].W.set_mask(m1)   # hidden unit 11: no outputs -> dropped
    m0 = lin[0].W.mask.copy(); m0[:6, :] = False; lin[0].W.set_mask(m0)   # inputs 0..5: never read
    return m


def test_sparse_mlp_paths_match_the_masked_dense_model_exactly():
    m = _pruned_mlp()
    x = np.random.default_rng(1).random((64, 50))
    with no_grad():
        want = m(Tensor(x)).data
    s = SparseMLP([l.W.data for l in m.linears()], [l.W.mask for l in m.linears()], [l.b.data for l in m.linears()])
    np.testing.assert_allclose(s.forward_compact_dense(x), want, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(s.forward_csr(x), want, rtol=1e-10, atol=1e-10)


def test_compaction_removes_dead_inputs_and_units_and_counts_work():
    m = _pruned_mlp()
    lin = m.linears()
    s = SparseMLP([l.W.data for l in lin], [l.W.mask for l in lin], [l.b.data for l in lin])
    c = s.cost()
    assert c["inputs_read"] <= 50 - 6
    assert c["compact_shapes"][0][1] < 40 and c["compact_shapes"][1][0] == c["compact_shapes"][0][1]
    live = sum(int(l.W.mask.sum()) for l in lin)
    assert c["macs_per_sample"]["csr"] <= live          # removing dead units can only drop more
    assert c["macs_per_sample"]["dense"] == 50 * 40 + 40 * 30 + 30 * 5
    assert c["macs_per_sample"]["csr"] <= c["macs_per_sample"]["compact_dense"] <= c["macs_per_sample"]["dense"]
