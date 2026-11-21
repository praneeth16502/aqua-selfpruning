"""Part 1 correctness requirement: masked weights.

"The gradient of a masked weight", in one sentence:
    p.grad = mask * dL/dw_eff is the true derivative w.r.t. the stored weight (0 where pruned) and is the
    ONLY thing the optimizer sees; p.dense_grad = dL/dw_eff (pre-mask) is a separate, named signal used for
    saliency/regrowth and never reaches the optimizer.
"""
import numpy as np

from engine import Tensor, numerical_grad, softmax_cross_entropy
from engine.parameter import Parameter
from nn.optim import Adam

rng = np.random.default_rng(42)
N, D, K = 6, 5, 4
X = rng.normal(size=(N, D))
Y = rng.integers(0, K, size=N)
MASK = rng.random((D, K)) < 0.5
MASK[0, 0], MASK[0, 1] = True, False  # guarantee both kinds exist
W0 = rng.normal(size=(D, K))


def loss_of(p: Parameter) -> Tensor:
    return softmax_cross_entropy(Tensor(X) @ p.masked(), Y)


# ----------------------------------------------------------------- (a) forward
def test_a_masked_weight_contributes_exactly_zero_in_forward():
    outs = []
    for poison in (1e30, np.inf, np.nan):  # poison pruned slots: none may leak into the forward pass
        p = Parameter(W0.copy(), mask=MASK)
        p.data[~MASK] = poison  # bypass set_mask on purpose: simulate a corrupted stored value
        eff = p.masked()
        assert np.array_equal(eff.data[~MASK], np.zeros((~MASK).sum())), "masked weights must be exact 0.0"
        outs.append((Tensor(X) @ eff).data)
    for o in outs[1:]:
        assert np.array_equal(o, outs[0]), "output changed with the value stored in a pruned slot"
    # independent reference: sum over ACTIVE connections only, written as an explicit loop
    ref = np.zeros((N, K))
    for k in range(K):
        for d in range(D):
            if MASK[d, k]:
                ref[:, k] += X[:, d] * W0[d, k]
    np.testing.assert_allclose(outs[0], ref, rtol=1e-12, atol=1e-14)


# ------------------------------------------------------ (b) 0 grad for optimizer
def test_b_masked_weight_gets_exactly_zero_update_grad_but_nonzero_signal_grad():
    p = Parameter(W0.copy(), mask=MASK)
    loss_of(p).backward()
    assert np.all(p.grad[~MASK] == 0.0)
    assert np.all(p.dense_grad[~MASK] != 0.0), "regrowth signal must survive masking (else regrowth is impossible)"
    assert np.array_equal(p.grad, np.where(MASK, p.dense_grad, 0.0))  # the definition, exactly


# -------------------------------------------- (c) unmasked weights correct grad
def test_c_update_grad_matches_finite_differences_everywhere_including_masked():
    p = Parameter(W0.copy(), mask=MASK)
    loss_of(p).backward()

    def f_stored(w):  # L as a function of the STORED weights
        return float(loss_of(Parameter(w, mask=MASK)).data)

    num = numerical_grad(f_stored, p.data)
    np.testing.assert_allclose(p.grad, num, rtol=1e-6, atol=1e-8)
    assert np.all(num[~MASK] == 0.0)  # true derivative w.r.t. a pruned stored weight is 0 (not approx)


def test_c2_signal_grad_matches_finite_differences_wrt_w_eff():
    p = Parameter(W0.copy(), mask=MASK)
    loss_of(p).backward()
    w_eff = np.where(MASK, p.data, 0.0)

    def f_eff(w):  # L as a function of the EFFECTIVE weights, dense, no mask in the graph
        return float(softmax_cross_entropy(Tensor(X) @ Tensor(w), Y).data)

    np.testing.assert_allclose(p.dense_grad, numerical_grad(f_eff, w_eff), rtol=1e-6, atol=1e-8)


# ----------------------------------------------------------- (d) optimizer step
def test_d1_pruned_weights_stay_exactly_zero_and_state_is_dropped_over_many_steps():
    p = Parameter(W0.copy(), mask=MASK)
    opt = Adam([p], lr=1e-2)
    for _ in range(25):
        opt.zero_grad()
        loss_of(p).backward()
        assert np.any(p.dense_grad[~MASK] != 0.0)  # pressure exists on pruned slots...
        opt.step()
        assert np.all(p.data[~MASK] == 0.0)  # ...and it never leaks into the weights
        assert np.all(opt.m[0][~MASK] == 0.0) and np.all(opt.v[0][~MASK] == 0.0) and np.all(opt.t[0][~MASK] == 0)
    assert np.any(p.data[MASK] != W0[MASK])  # active weights did train


def test_d2_adam_matches_textbook_reference_when_nothing_is_masked():
    w0, lr, b1, b2, eps = rng.normal(size=(3, 2)), 1e-2, 0.9, 0.999, 1e-8
    grads = [rng.normal(size=(3, 2)) for _ in range(10)]
    p = Parameter(w0.copy())
    opt = Adam([p], lr=lr)
    w, m, v = w0.copy(), np.zeros_like(w0), np.zeros_like(w0)
    for i, g in enumerate(grads, start=1):
        p.grad = g.copy()
        opt.step()
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        w = w - lr * (m / (1 - b1**i)) / (np.sqrt(v / (1 - b2**i)) + eps)
        np.testing.assert_allclose(p.data, w, rtol=1e-12, atol=1e-14)


def _revival_scenario(reset: bool, between_steps_only: bool = False) -> float:
    """1 weight. 50 steps of a strong POSITIVE gradient (builds big m, v) -> prune -> revive.
    After revival the true gradient is tiny and NEGATIVE, so descent must move w UP by ~lr.
    Returns the weight after the first post-revival step (it was revived at exactly 0)."""
    lr = 1e-2
    p = Parameter(np.array([1.0]))
    opt = Adam([p], lr=lr, reset_state_on_mask_change=reset)
    for _ in range(50):
        p.grad = np.array([10.0])
        opt.step()
    if between_steps_only:  # prune AND revive without any optimizer step in between
        p.set_mask(np.array([False]))
        p.set_mask(np.array([True]))
    else:
        p.set_mask(np.array([False]))
        for _ in range(5):  # masked steps: grad is exactly 0
            p.grad = np.array([0.0])
            opt.step()
            assert p.data[0] == 0.0
        p.set_mask(np.array([True]))  # revive at 0.0
    p.grad = np.array([-0.01])
    opt.step()
    return float(p.data[0])


def test_d3_revived_weight_moves_by_about_lr_in_the_current_descent_direction():
    lr = 1e-2
    w = _revival_scenario(reset=True)
    assert 0.99 * lr <= w <= lr * (1 + 1e-9), w  # bias-corrected fresh moments: |step| == lr, sign = -sign(g)


def test_d4_stale_moments_ablation_moves_the_revived_weight_UPHILL():
    w = _revival_scenario(reset=False)
    assert w < 0.0, w  # stale positive momentum overrides the current negative gradient: wrong direction
    assert abs(w) > 0.3e-2  # and not by a small amount (documents the corruption the policy prevents)


def test_d5_prune_then_revive_between_steps_still_gets_clean_state():
    lr = 1e-2
    w = _revival_scenario(reset=True, between_steps_only=True)
    assert 0.99 * lr <= w <= lr * (1 + 1e-9), w  # consume_mask_changes() catches the XOR-invisible round trip


def test_d6_revive_value_is_applied_and_pruned_value_is_zeroed():
    p = Parameter(np.array([[1.0, 2.0], [3.0, 4.0]]))
    p.set_mask(np.array([[True, False], [True, True]]))
    assert p.data[0, 1] == 0.0
    p.set_mask(np.array([[True, True], [True, True]]), revive_value=0.05)
    assert p.data[0, 1] == 0.05 and p.data[0, 0] == 1.0


def test_d7_in_flight_graph_uses_the_mask_snapshot_taken_at_forward_time():
    p = Parameter(W0.copy(), mask=MASK)
    loss = loss_of(p)
    p.set_mask(np.ones_like(MASK))  # mask changes between forward and backward
    loss.backward()
    assert np.all(p.grad[~MASK] == 0.0)


def test_d8_mask_shape_mismatch_rejected():
    p = Parameter(np.zeros((2, 2)))
    try:
        p.set_mask(np.ones((3, 3), dtype=bool))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_d9_optimizer_re_zeroes_a_pruned_slot_even_if_it_was_corrupted():
    """Guard invariant, tested in isolation: even if something wrote into a pruned slot (bug elsewhere,
    stale-state ablation, future code), the step must leave it exactly 0."""
    for reset in (True, False):
        p = Parameter(W0.copy(), mask=MASK)
        opt = Adam([p], lr=1e-2, reset_state_on_mask_change=reset)
        p.data[~MASK] = 0.7  # corrupt, bypassing set_mask
        p.grad = np.where(MASK, 0.3, 0.0)
        opt.step()
        assert np.all(p.data[~MASK] == 0.0), f"reset={reset}"
