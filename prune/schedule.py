"""Gradual sparsity schedule.

Cubic ramp (Zhu & Gupta style):

    s(t) = 0                                   t <= t0
    s(t) = s_f * (1 - (1 - (t - t0)/(t1 - t0))^3)   t0 < t < t1
    s(t) = s_f                                 t >= t1

Why cubic: ds/dt = 3 s_f (1-u)^2 / (t1-t0) is largest at the start, when the network is most
redundant and the cheapest weights are genuinely near-useless, and goes to 0 at the end, when every
remaining weight matters and the network needs time between cuts to re-route.  A linear ramp cuts as
hard at 85% -> 90% as at 0% -> 5%; one-shot pruning at the end gives the network no time to adapt at all.
"""
from __future__ import annotations


def cubic_sparsity(step: int, start: int, end: int, final: float) -> float:
    if not 0.0 <= final < 1.0:
        raise ValueError(f"final sparsity must be in [0, 1), got {final}")
    if end <= start:
        raise ValueError("end must be > start")
    if step <= start:
        return 0.0
    if step >= end:
        return final
    u = (step - start) / (end - start)
    return final * (1.0 - (1.0 - u) ** 3)
