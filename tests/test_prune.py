"""Step 3: schedule, allocation, hard budget, honest masking, and criterion behaviour."""
import numpy as np

from engine import Parameter
from nn.layers import MLP
from nn.optim import Adam
from prune import Magnitude, PruneManager, Saliency, allocate_keep, cubic_sparsity
from train.data import make_split
from train.loop import evaluate, train


# ---------------------------------------------------------------- schedule
def test_cubic_endpoints_and_shape():
    assert cubic_sparsity(0, 10, 110, 0.9) == 0.0
    assert cubic_sparsity(10, 10, 110, 0.9) == 0.0
    assert cubic_sparsity(110, 10, 110, 0.9) == 0.9
    assert cubic_sparsity(10_000, 10, 110, 0.9) == 0.9
    s = np.array([cubic_sparsity(t, 10, 110, 0.9) for t in range(10, 111)])
    d = np.diff(s)
    assert np.all(d >= 0)                     # monotone
    assert np.all(np.diff(d) <= 1e-12)        # concave: cuts hardest early, gently late
    assert abs(cubic_sparsity(60, 10, 110, 0.9) - 0.9 * (1 - 0.5 ** 3)) < 1e-12


def test_cubic_rejects_bad_args():
    for args in [(0, 0, 0, 0.5), (0, 0, 10, 1.0), (0, 0, 10, -0.1)]:
        try:
            cubic_sparsity(*args)
        except ValueError:
            continue
        raise AssertionError(f"accepted {args}")


# ---------------------------------------------------------------- allocation
def test_allocate_keeps_exact_count_top_scores_and_floors():
    r = np.random.default_rng(0)
    scores = [r.random((20, 10)), r.random((10, 3)) * 1e-3]  # layer 2 globally "worse" everywhere
    live = [np.ones((20, 10), bool), np.ones((10, 3), bool)]
    new = allocate_keep(scores, live, keep_total=40, min_keep=[0, 5])
    assert sum(int(m.sum()) for m in new) == 40
    assert new[1].sum() == 5                                   # floor rescued the small layer
    assert set(np.flatnonzero(new[1])) == set(np.argsort(-scores[1].ravel())[:5])
    kept0 = np.sort(scores[0][new[0]])
    dropped0 = scores[0][~new[0]]
    assert kept0.min() >= dropped0.max()                       # global part is a true top-k


def test_allocate_never_revives_and_rejects_impossible_budgets():
    s = [np.arange(12.0).reshape(3, 4)]
    live = [np.zeros((3, 4), bool)]
    live[0][0, :2] = True                                      # only 2 live, and they have the LOWEST scores
    new = allocate_keep(s, live, 2, [0])
    assert np.array_equal(new[0], live[0])
    for keep, floor in [(3, [0]), (1, [2])]:
        try:
            allocate_keep(s, live, keep, floor)
        except ValueError:
            continue
        raise AssertionError("impossible budget accepted")


# ---------------------------------------------------------------- criteria
def test_saliency_is_ema_of_abs_w_times_dense_grad_and_needs_warmup():
    p = Parameter(np.array([[1.0, -2.0]]))
    sal = Saliency(decay=0.5)
    try:
        sal.scores([p])
    except RuntimeError:
        pass
    else:
        raise AssertionError("scored with no observations")
    for g in ([1.0, 1.0], [-1.0, 3.0]):        # signed gradients that cancel for w0
        p.dense_grad = np.array([g])
        sal.observe([p])
    # raw EMA: 0.5*(0.5*|w g1|) + 0.5*|w g2| ; bias correction 1 - 0.5^2
    want = (0.25 * np.abs(p.data * [1.0, 1.0]) + 0.5 * np.abs(p.data * [-1.0, 3.0])) / 0.75
    np.testing.assert_allclose(sal.scores([p])[0], want)
    assert sal.scores([p])[0][0, 0] > 0        # |w * mean g| would be 0 for w0: the reason we average |w g|


def test_saliency_uses_dense_grad_not_masked_grad():
    p = Parameter(np.ones((1, 2)), mask=np.array([[True, False]]))
    p.grad = np.array([[1.0, 0.0]])
    p.dense_grad = np.array([[1.0, 5.0]])
    sal = Saliency(decay=0.0)
    sal.observe([p])
    # pruned entry: stored w is 0 so its saliency is 0 regardless; live entry uses dense_grad
    np.testing.assert_allclose(sal.scores([p])[0], [[1.0, 0.0]])


def _dead_input_data(n=800, d=12, n_dead=6, seed=0):
    """Features [0, n_dead) are always exactly 0 (like MNIST border pixels); the rest carry the label."""
    r = np.random.default_rng(seed)
    y = r.integers(0, 3, n)
    X = r.normal(0, 1, (n, d))
    X[:, n_dead:] += 2.0 * np.eye(3)[y] @ r.normal(0, 1, (3, d - n_dead))
    X[:, :n_dead] = 0.0
    X = (X - X.min()) / (X.max() - X.min())
    X[:, :n_dead] = 0.0
    return make_split(X, y, 1.0, n_test=100, n_val=100, seed=0)


def test_saliency_prunes_dead_inputs_first_magnitude_does_not():
    """Weights on an always-zero input get exactly zero gradient, so Adam never moves them: they keep
    their random init magnitude.  Saliency scores them exactly 0; magnitude keeps the large ones.

    Dead ReLU units (never active) also give their weights exactly zero saliency, which saliency is right
    about too.  So the budget here = the number of zero-saliency weights; with that budget saliency must remove
    every dead-input weight, and magnitude (same model, same budget) must keep some of them."""
    data = _dead_input_data()
    n_dead = 6
    m = MLP([12, 16, 3], np.random.default_rng(1))
    params = m.prunable()
    sal = Saliency()
    train(m, data, epochs=3, batch_size=32, lr=1e-2, seed=0, log=None, on_backward=lambda s: sal.observe(params))
    assert np.all(params[0].dense_grad[:n_dead] == 0)
    total = sum(p.data.size for p in params)
    n_zero = sum(int((s == 0).sum()) for s in sal.scores(params))
    assert n_zero >= n_dead * 16
    snapshot = [p.data.copy() for p in params]
    kept_dead = {}
    for crit in (sal, Magnitude()):
        for p, w in zip(params, snapshot):
            p.set_mask(np.ones_like(p.mask))
            p.data[...] = w
        mgr = PruneManager(params, crit, final_sparsity=0.5, start_step=0, end_step=1, min_density=0.0)
        mgr.prune_to(n_zero / total)
        assert mgr.active() == total - n_zero
        kept_dead[crit.name] = int(params[0].mask[:n_dead].sum())
    assert kept_dead["saliency"] == 0, kept_dead
    assert kept_dead["magnitude"] > 0, kept_dead


# ---------------------------------------------------------------- end-to-end hard budget + honest masking
def _blobs(n=600, d=10, k=3, seed=0):
    r = np.random.default_rng(seed)
    y = r.integers(0, k, n)
    X = r.normal(0, 2.0, (k, d))[y] + r.normal(0, 1.0, (n, d))
    X = (X - X.min()) / (X.max() - X.min())
    return make_split(X, y, 1.0, n_test=100, n_val=100, seed=0)


def test_manager_hits_budget_exactly_and_masks_are_real_zeros_through_finetune():
    data = _blobs()
    m = MLP([10, 64, 32, 3], np.random.default_rng(0))
    params = m.prunable()
    total = sum(p.data.size for p in params)
    opt = Adam(m.parameters(), lr=1e-2)
    mgr = PruneManager(params, Saliency(), final_sparsity=0.9, start_step=15, end_step=75, every=10, min_density=0.05)
    seen = []

    def on_step(step):
        mgr.on_step(step)
        if step > mgr.end:  # fine-tune phase: budget must hold and pruned weights stay exact zeros
            seen.append(mgr.active())
            for i, p in enumerate(params):
                assert np.all(p.data[~p.mask] == 0.0)
                j = opt.params.index(p)
                assert np.all(opt.m[j][~p.mask] == 0) and np.all(opt.v[j][~p.mask] == 0)

    h = train(m, data, epochs=12, batch_size=32, lr=1e-2, seed=0, log=None, opt=opt,
              on_backward=mgr.on_backward, on_step=on_step)
    assert mgr.active() == total - round(0.9 * total)
    assert seen and len(set(seen)) == 1                     # frozen after the ramp
    ach = [e["achieved"] for e in mgr.events]
    assert all(b >= a for a, b in zip(ach, ach[1:]))         # monotone
    assert all(abs(e["achieved"] - e["target"]) <= 1.0 / total + 1e-12 for e in mgr.events)
    assert all(int(p.mask.sum()) >= int(0.05 * p.data.size) for p in params)  # 5% floor (rounded down) per layer
    assert np.isfinite(h["train_loss"]).all()
    _, acc = evaluate(m, data.x_test, data.y_test)
    assert acc > 0.85


def test_zero_target_is_exactly_dense_training():
    data = _blobs()
    runs = []
    for prune in (False, True):
        m = MLP([10, 16, 3], np.random.default_rng(3))
        kw = {}
        if prune:
            mgr = PruneManager(m.prunable(), Saliency(), 0.0, 5, 20, every=5)
            kw = dict(on_backward=mgr.on_backward, on_step=mgr.on_step)
        runs.append(train(m, data, 3, 32, 1e-2, seed=3, log=None, **kw)["train_loss"])
    np.testing.assert_array_equal(runs[0], runs[1])


def test_fisher_is_w_squared_times_ema_of_g_squared_and_survives_cancelling_gradients():
    from prune import Fisher
    p = Parameter(np.array([[1.0, -2.0]]))
    f = Fisher(decay=0.5)
    for g in ([1.0, 1.0], [-1.0, 3.0]):
        p.dense_grad = np.array([g])
        f.observe([p])
    ema = (0.25 * np.array([1.0, 1.0]) + 0.5 * np.array([1.0, 9.0])) / 0.75
    np.testing.assert_allclose(f.scores([p])[0], p.data ** 2 * ema)
    assert f.scores([p])[0][0, 0] > 0           # mean g for w0 is 0, but E[g^2] is not


# ---------------------------------------------------------------- regrowth
def _pruned_manager(grace=None, regrow_frac=0.5):
    from prune import Fisher
    p = Parameter(np.arange(1.0, 41.0).reshape(4, 10))   # bigger index = bigger weight = higher fisher score
    mgr = PruneManager([p], Fisher(decay=0.0), final_sparsity=0.5, start_step=0, end_step=10, every=10,
                       min_density=0.0, regrow=True, regrow_every=5, regrow_frac=regrow_frac, regrow_min=0.0,
                       grace=grace)
    p.dense_grad = np.ones_like(p.data)
    mgr.on_backward(0)
    mgr.prune_to(0.5, step=10)                             # keeps the 20 largest (indices 20..39)
    return p, mgr


def test_regrow_swap_keeps_count_and_revives_by_dense_grad():
    p, mgr = _pruned_manager()
    live0 = mgr.active()
    g = np.zeros_like(p.data)
    g.ravel()[[3, 7]] = 5.0                                # two dead weights with large gradient signal
    p.dense_grad = g
    mgr.on_backward(11)
    k = mgr.regrow_swap(step=14)
    assert k == 10                                         # 50% of the 20 pruned since the last swap
    assert mgr.active() == live0                           # a swap, not extra capacity
    assert p.mask.ravel()[3] and p.mask.ravel()[7]         # highest |dense_grad| came back first
    assert np.all(p.data[p.mask & mgr.ever_revived[0]] == 0.0)   # revived at exactly 0
    assert mgr.regrowth_stats()["unique_revived"] == 10


def test_grace_period_protects_revived_weights_and_without_it_they_die_immediately():
    for grace, survive in ((5, True), (0, False)):
        p, mgr = _pruned_manager(grace=grace, regrow_frac=0.05)   # k = 1: only the planted weight comes back
        g = np.zeros_like(p.data)
        g.ravel()[3] = 5.0
        p.dense_grad = g
        mgr.on_backward(11)
        mgr.regrow_swap(step=14)
        assert p.mask.ravel()[3]
        p.dense_grad = np.ones_like(p.data)
        mgr.on_backward(12)
        mgr.prune_to(0.6, step=15)                         # revived weight has w=0 -> fisher score 0
        assert bool(p.mask.ravel()[3]) == survive, (grace, survive)


def test_regrowth_end_to_end_keeps_exact_budget_and_floors():
    from prune import Fisher
    data = _blobs()
    m = MLP([10, 64, 32, 3], np.random.default_rng(0))
    params = m.prunable()
    total = sum(p.data.size for p in params)
    mgr = PruneManager(params, Fisher(), 0.9, start_step=15, end_step=75, every=10, min_density=0.05,
                       regrow=True, regrow_every=15)
    train(m, data, epochs=12, batch_size=32, lr=1e-2, seed=0, log=None,
          on_backward=mgr.on_backward, on_step=mgr.on_step)
    assert mgr.active() == total - round(0.9 * total)
    assert all(e["sparsity"] >= 0.9 - 1e-12 or e["step"] < mgr.end for e in mgr.regrow_events)
    assert len(mgr.regrow_events) > 5 and mgr.regrowth_stats()["unique_revived"] > 0
    assert all(int(p.mask.sum()) >= int(0.05 * p.data.size) for p in params)
    for p in params:
        assert np.all(p.data[~p.mask] == 0.0)


def test_floor_never_exceeds_budget_and_does_not_force_uniform_layers_at_high_sparsity():
    """Regression: at 95% with a 5% floor, ceil-rounded floors summed to budget + 1 (crash), and even exact floors
    would have forced every layer to exactly 5%.  Floor is now min(min_density, (1-s)/2), rounded down."""
    from prune import Fisher
    r = np.random.default_rng(0)
    ps = [Parameter(r.normal(size=s)) for s in [(784, 256), (256, 128), (128, 10)]]
    for i, p in enumerate(ps):
        p.dense_grad = np.full_like(p.data, 10.0 ** -i)      # layers clearly ranked: layer 0 most salient
    f = Fisher(decay=0.0)
    f.observe(ps)
    for s in (0.9, 0.95, 0.98):
        for p in ps:
            p.set_mask(np.ones_like(p.mask))
        mgr = PruneManager(ps, f, s, 0, 10, min_density=0.05)
        mgr.prune_to(s)
        assert mgr.active() == mgr.total - round(s * mgr.total)
        dens = [1 - x for x in mgr.layer_sparsity()]
        assert all(d >= min(0.05, 0.5 * (1 - s)) - 1e-3 for d in dens)
        assert max(dens) - min(dens) > 0.01                  # global ranking still decides, not the floor
