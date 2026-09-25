"""Small exact/approximate statistical functions (no SciPy needed)."""

from __future__ import annotations

import math


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    logs = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        + i * math.log(p) + (n - i) * math.log1p(-p)
        for i in range(k, n + 1)
    ]
    m = max(logs)
    return min(1.0, math.exp(m) * sum(math.exp(v - m) for v in logs))


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def _gammainc_lower_reg(a: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x < a + 1:  # series
        term = 1.0 / a
        total = term
        ap = a
        for _ in range(1000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # continued fraction for the upper part
    b = x + 1 - a
    c = 1e300
    d = 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x: float, dof: int) -> float:
    """P(Chi2(dof) >= x)."""
    if dof <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - _gammainc_lower_reg(dof / 2.0, x / 2.0)))
