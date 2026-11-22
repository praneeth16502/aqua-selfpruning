"""Layers, init, masking wired through a whole model, and the float64-vs-float32 separation."""
import numpy as np

from engine import Tensor, numerical_grad, softmax_cross_entropy
from nn.layers import MLP, Linear
from nn.optim import Adam
from train.init_study import second_moments


def _batch(n=8, d=6, k=3, seed=0):
    r = np.random.default_rng(seed)
    return r.normal(size=(n, d)), r.integers(0, k, size=n)


def _loss(net, x, y):
    return softmax_cross_entropy(net(Tensor(x)), y)


# ------------------------------------------------ float64: finite-difference check of the WHOLE model
def test_mlp_all_param_grads_match_finite_differences_with_masks_active():
    x, y = _batch()
    rng = np.random.default_rng(1)
    net = MLP([6, 9, 7, 3], rng, dtype=np.float64)
    for W in net.prunable():  # prune ~50% of every weight matrix
        W.set_mask(rng.random(W.data.shape) < 0.5)
    for l in net.linears():  # nonzero biases so their grads are non-trivial; keep ReLU away from its kink
        l.b.data[...] = rng.normal(0, 0.3, l.b.data.shape)
    net.zero_grad()
    _loss(net, x, y).backward()
    for p in net.parameters():
        def f(arr, p=p):
            old = p.data.copy()
            p.data[...] = arr
            try:
                return float(_loss(net, x, y).data)
            finally:
                p.data[...] = old
        np.testing.assert_allclose(p.grad, numerical_grad(f, p.data), rtol=1e-6, atol=1e-8)
    for W in net.prunable():
        assert np.all(W.grad[~W.mask] == 0.0)


# ------------------------------------------------ float32: SEPARATE check, looser, against the float64 reference
def test_float32_model_stays_float32_and_agrees_with_float64_reference():
    x, y = _batch(n=32, d=20, k=5, seed=3)
    nets = {dt: MLP([20, 64, 32, 5], np.random.default_rng(7), dtype=dt) for dt in (np.float32, np.float64)}
    grads, logits = {}, {}
    for dt, net in nets.items():
        out = net(Tensor(x.astype(dt)))
        net.zero_grad()
        softmax_cross_entropy(out, y).backward()
        grads[dt] = [p.grad for p in net.parameters()]
        logits[dt] = out.data
    assert logits[np.float32].dtype == np.float32, "silent upcast in forward"
    assert all(g.dtype == np.float32 for g in grads[np.float32]), "silent upcast in backward"
    for g32, g64 in zip(grads[np.float32], grads[np.float64]):  # NOT the 1e-6 finite-difference tolerance
        rel = np.linalg.norm(g32 - g64) / (np.linalg.norm(g64) + 1e-12)
        assert rel < 1e-4, rel


def test_adam_keeps_float32_params_and_state_float32():
    net = MLP([6, 8, 3], np.random.default_rng(0), dtype=np.float32)
    opt = Adam(net.parameters(), lr=1e-3)
    x, y = _batch()
    for _ in range(3):
        opt.zero_grad()
        _loss(net, x.astype(np.float32), y).backward()
        opt.step()
    assert all(p.data.dtype == np.float32 for p in net.parameters())
    assert all(m.dtype == np.float32 for m in opt.m) and all(v.dtype == np.float32 for v in opt.v)


# ------------------------------------------------ init
def test_he_init_preserves_second_moment_with_depth_and_alternatives_do_not():
    he = second_moments(2.0)
    xav = second_moments(1.0)
    tiny, big = second_moments(2.0, std_override=0.01), second_moments(2.0, std_override=0.1)
    assert max(he) / min(he) < 1.6                       # flat across 8 layers
    assert xav[-1] / xav[0] < 0.02                       # ~2^-L decay under ReLU
    assert tiny[-1] < 1e-9 and big[-1] > 1e2            # vanish / explode


def test_init_std_matches_theory_and_biases_zero():
    rng = np.random.default_rng(0)
    l = Linear(784, 256, rng, gain=2.0, dtype=np.float64)
    assert abs(l.W.data.std() / np.sqrt(2 / 784) - 1) < 0.02
    assert np.all(l.b.data == 0)


def test_initial_loss_is_near_log_num_classes_so_logits_start_unbiased():
    rng = np.random.default_rng(0)
    net = MLP([784, 256, 128, 10], rng, dtype=np.float32)
    x = np.random.default_rng(1).random((512, 784)).astype(np.float32)
    y = np.random.default_rng(2).integers(0, 10, 512)
    loss = float(_loss(net, x, y).data)
    assert abs(loss - np.log(10)) < 0.5, loss  # gain-1 logits layer: near-uniform softmax at init


def test_prunable_params_are_only_weight_matrices_and_most_params():
    net = MLP([784, 256, 128, 10], np.random.default_rng(0))
    w = sum(p.data.size for p in net.prunable())
    tot = sum(p.data.size for p in net.parameters())
    assert w == 784 * 256 + 256 * 128 + 128 * 10 and w / tot > 0.99
