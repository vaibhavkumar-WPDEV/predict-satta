"""Self-correcting ensemble.

Every day the final prediction is a weighted mixture of the experts:

    P_t(v) = Σ_i w_(t,i) · P_(t,i)(v)

After the real result y_t is known every weight is multiplied by how much
probability that expert gave to the real number (Bayes / Hedge update) and a
small share alpha is spread back to all experts (Fixed-Share, Herbster &
Warmuth 1998) so a model that was wrong for a while can come back:

    w'_(t+1,i) ∝ w_(t,i) · P_(t,i)(y_t)^eta
    w_(t+1,i)  = (1-alpha)·w'_(t+1,i) + alpha/N

Guarantee (for alpha = 0, eta = 1):  L_mix ≤ min_i L_i + ln N,
where L is the cumulative log-loss. So the mixture is never much worse than
the best single expert in hindsight; the dashboard checks this on real data.

Self-tuning (meta-learning): how fast to learn (eta) and how much to forget
(alpha) are not fixed guesses. Nine ensembles run side by side, one for each
eta in {0.5, 1, 2} and alpha in {0.002, 0.01, 0.05}, and a top-level Hedge
weights them by how well each predicted. Because every ensemble is linear in
the experts, the final prediction is again one expert mixture with effective
weights  w_eff = Σ_g v_g · w_g.

Merge layer: the experts' opinions are merged two ways, a linear pool
Σ w_i P_i (keeps every expert's doubts) and a geometric pool ∝ Π P_i^(w_i)
(sharpens where the experts agree). A last Hedge learns how much of each to
use, so the tool decides from results which way of merging works better.

replay() walks forward through history and at each day uses only the data
before that day: the backtest is exactly what the live system would have
predicted on that day.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np

from .. import config
from .base import SeriesData, andar_bahar, cut, rev
from .experts import Expert, default_experts

GRID = [(eta, alpha) for eta in (0.5, 1.0, 2.0) for alpha in (0.002, 0.01, 0.05)]
META_ALPHA = 0.01


@dataclass
class Step:
    date: dt.date
    actual: int
    mix: np.ndarray            # final distribution used on that day
    weights: np.ndarray        # weights before the update
    new_weights: np.ndarray    # weights after learning from this result
    p_actual: np.ndarray       # each expert's probability of the actual number
    ranks: np.ndarray          # each expert's rank of the actual number (1 = best)
    equal: np.ndarray          # same experts, equal weights, no learning (control)


@dataclass
class Replay:
    market: str
    experts: list[Expert]
    steps: list[Step] = field(default_factory=list)
    weights: np.ndarray | None = None
    losses: np.ndarray | None = None    # cumulative log-loss per expert
    mix_loss: float = 0.0
    grid: list = field(default_factory=list)
    meta: np.ndarray | None = None      # weight of each (eta, alpha) ensemble
    merge: np.ndarray | None = None     # weight of the linear and the geometric pool
    correction: np.ndarray | None = None  # weight of each miss-correction shift


def rank_of(p: np.ndarray, v: int) -> int:
    """1-based position of v when numbers are sorted by probability (ties by number)."""
    order = np.argsort(-p, kind="stable")
    return int(np.where(order == v)[0][0]) + 1


def expert_matrix(sd: SeriesData, experts: list[Expert], i: int, date: dt.date | None = None) -> np.ndarray:
    ctx = sd.context(i, date)
    return np.stack([e.predict(ctx) for e in experts])


# Miss-correction: if results keep landing on a fixed transform of the numbers
# we rank high (their palti, ±1, ±10, ±11 or cut), the whole list is moved that
# way. P_shift(v) = P(T⁻¹(v)); a Hedge over the shifts learns from every result
# where the real number fell relative to our list. "same" = no correction.
SHIFTS = {
    "same": lambda x: x,
    "palti": rev,
    "+1": lambda x: (x + 1) % 100,
    "-1": lambda x: (x - 1) % 100,
    "+10": lambda x: (x + 10) % 100,
    "-10": lambda x: (x - 10) % 100,
    "+11": lambda x: (x + 11) % 100,
    "-11": lambda x: (x - 11) % 100,
    "cut": cut,
}
_V = np.arange(100)
# SHIFT_INV[k][v] = the number that shift k moves onto v
SHIFT_INV = np.stack([np.argsort(fn(_V)) for fn in SHIFTS.values()])
SHIFT_PRIOR = np.array([0.6] + [0.4 / (len(SHIFTS) - 1)] * (len(SHIFTS) - 1))


def pools(w: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear pool Σ w_i P_i and geometric pool ∝ exp(Σ w_i log P_i)."""
    lin = w @ P
    lg = w @ np.log(np.maximum(P, 1e-12))
    geo = np.exp(lg - lg.max())
    return lin, geo / geo.sum()


def replay(sd: SeriesData, experts: list[Expert] | None = None, start: int | None = None,
           grid: list[tuple[float, float]] | None = None) -> Replay:
    experts = experts or default_experts()
    start = config.MIN_TRAIN if start is None else start
    grid = grid or GRID
    k, g = len(experts), len(grid)
    etas = np.array([e for e, _ in grid])[:, None]
    alphas = np.array([a for _, a in grid])[:, None]
    W = np.full((g, k), 1.0 / k)   # expert weights inside each ensemble
    v = np.full(g, 1.0 / g)        # meta weights over the ensembles
    losses = np.zeros(k)
    u = np.array([0.5, 0.5])        # linear vs geometric merge
    c = SHIFT_PRIOR.copy()          # miss-correction: how the result moves from our list
    rep = Replay(sd.market, experts, grid=grid)
    for i in range(start, sd.n):
        P = expert_matrix(sd, experts, i)
        w = v @ W
        lin, geo = pools(w, P)
        merged = u[0] * lin + u[1] * geo
        shifted = merged[SHIFT_INV]                 # (K, 100): list moved by each shift
        mix = c @ shifted
        y = int(sd.y[i])
        u = u * np.array([lin[y], geo[y]])
        u /= u.sum()
        u = (1 - META_ALPHA) * u + META_ALPHA / 2
        c = c * shifted[:, y]
        c /= c.sum()
        c = (1 - META_ALPHA) * c + META_ALPHA / len(c)
        pa = np.maximum(P[:, y], 1e-300)
        losses -= np.log(pa)
        rep.mix_loss -= math.log(max(float(mix[y]), 1e-300))
        v = v * (W @ pa)
        v /= v.sum()
        v = (1 - META_ALPHA) * v + META_ALPHA / g
        W = W * np.power(pa[None, :], etas)
        W /= W.sum(axis=1, keepdims=True)
        W = (1 - alphas) * W + alphas / k
        nw = v @ W
        order = np.argsort(-P, axis=1, kind="stable")
        ranks = np.argmax(order == y, axis=1) + 1
        rep.steps.append(Step(sd.dates[i], y, mix, w, nw, pa, ranks, P.mean(axis=0)))
    rep.weights = v @ W
    rep.meta = v
    rep.merge = u
    rep.correction = c
    rep.losses = losses
    return rep


def predict_next(sd: SeriesData, rep: Replay, date: dt.date) -> tuple[np.ndarray, np.ndarray]:
    """(final distribution, expert matrix) for the next unknown draw on `date`."""
    P = expert_matrix(sd, rep.experts, sd.n, date)
    lin, geo = pools(rep.weights, P)
    u = rep.merge if rep.merge is not None else np.array([1.0, 0.0])
    c = rep.correction if rep.correction is not None else SHIFT_PRIOR
    return c @ (u[0] * lin + u[1] * geo)[SHIFT_INV], P


# ------------------------------------------------------------- summaries

def top_list(p: np.ndarray, k: int = 10) -> list[list]:
    order = np.argsort(-p, kind="stable")[:k]
    return [[int(v), round(float(p[v]), 5)] for v in order]


def digit_top(p10: np.ndarray, k: int = 3) -> list[list]:
    order = np.argsort(-p10, kind="stable")[:k]
    return [[int(d), round(float(p10[d]), 4)] for d in order]


def score(dist: np.ndarray, actual: int) -> dict:
    r = rank_of(dist, actual)
    at, ab = andar_bahar(dist)
    return {
        "rank": r,
        "hit1": r <= 1,
        "hit5": r <= 5,
        "hit10": r <= 10,
        "andar_hit": actual // 10 in [d for d, _ in digit_top(at)],
        "bahar_hit": actual % 10 in [d for d, _ in digit_top(ab)],
        "prob": float(dist[actual]),
        "logloss": -math.log(max(float(dist[actual]), 1e-300)),
    }


def postmortem(step: Step, experts: list[Expert], official: dict | None = None) -> dict:
    """Why the prediction hit or missed, and what the system learned (Hinglish)."""
    sc = official or score(step.mix, step.actual)
    a = step.actual
    lines = []
    if sc["hit10"]:
        lines.append(f"✔ HIT: {a:02d} hamari top-10 list me #{sc['rank']} par tha.")
    else:
        lines.append(f"✘ MISS: {a:02d} ko rank {sc['rank']}/100 mila (probability {sc['prob']:.2%}, "
                     f"random = 1.00%).")
    lines.append(("✔" if sc["andar_hit"] else "✘") + f" Andar {a // 10} top-3 me "
                 + ("tha." if sc["andar_hit"] else "nahi tha."))
    lines.append(("✔" if sc["bahar_hit"] else "✘") + f" Bahar {a % 10} top-3 me "
                 + ("tha." if sc["bahar_hit"] else "nahi tha."))
    top10 = [v for v, _ in top_list(step.mix)]
    near = [v for v in top10 if v in (int((a % 10) * 10 + a // 10), (a + 1) % 100, (a - 1) % 100)]
    if near and not sc["hit10"]:
        lines.append(f"Kareeb: top-10 me {', '.join(f'{v:02d}' for v in near)} tha (palti/±1).")
    best = int(np.argmin(step.ranks))
    lines.append(f"Aaj sabse sahi model: {experts[best].label} — isne {a:02d} ko rank "
                 f"{int(step.ranks[best])} diya.")
    blame = step.weights * np.maximum(0, 0.01 - step.p_actual)
    if blame.max() > 0:
        j = int(np.argmax(blame))
        lines.append(f"Galti ki wajah: {experts[j].label} par {step.weights[j]:.1%} bharosa tha, "
                     f"par isne {a:02d} ko sirf {step.p_actual[j]:.2%} diya.")
    delta = step.new_weights - step.weights
    up = int(np.argmax(delta))
    down = int(np.argmin(delta))
    lines.append(f"Sudhar (learning): {experts[up].label} ka weight {step.weights[up]:.1%} → "
                 f"{step.new_weights[up]:.1%}; {experts[down].label} {step.weights[down]:.1%} → "
                 f"{step.new_weights[down]:.1%}.")
    return {"lines": lines, "best_expert": experts[best].name}


def summarize(scores: list[dict]) -> dict:
    """Hit rates vs chance with exact binomial p-values."""
    from .stats import binom_sf, norm_sf

    n = len(scores)
    out = {"n": n}
    if n == 0:
        return out
    for key, p0 in (("hit1", 0.01), ("hit5", 0.05), ("hit10", 0.10), ("andar_hit", 0.30), ("bahar_hit", 0.30)):
        hits = sum(1 for s in scores if s[key])
        out[key] = {"hits": hits, "rate": hits / n, "chance": p0, "expected": p0 * n,
                    "p_value": binom_sf(hits, n, p0)}
    mean_rank = sum(s["rank"] for s in scores) / n
    z = (50.5 - mean_rank) / math.sqrt(833.25 / n)
    out["mean_rank"] = {"value": mean_rank, "chance": 50.5, "z": z, "p_value": norm_sf(z)}
    ll = sum(s["logloss"] for s in scores) / n
    out["logloss"] = {"value": ll, "chance": math.log(100), "skill": 1 - ll / math.log(100)}
    return out
