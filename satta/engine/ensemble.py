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
from .base import SeriesData, andar_bahar
from .experts import Expert, default_experts

ETA = 1.0
ALPHA = 0.01


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


def rank_of(p: np.ndarray, v: int) -> int:
    """1-based position of v when numbers are sorted by probability (ties by number)."""
    order = np.argsort(-p, kind="stable")
    return int(np.where(order == v)[0][0]) + 1


def expert_matrix(sd: SeriesData, experts: list[Expert], i: int, date: dt.date | None = None) -> np.ndarray:
    ctx = sd.context(i, date)
    return np.stack([e.predict(ctx) for e in experts])


def replay(sd: SeriesData, experts: list[Expert] | None = None, start: int | None = None,
           eta: float = ETA, alpha: float = ALPHA) -> Replay:
    experts = experts or default_experts()
    start = config.MIN_TRAIN if start is None else start
    k = len(experts)
    w = np.full(k, 1.0 / k)
    losses = np.zeros(k)
    rep = Replay(sd.market, experts)
    for i in range(start, sd.n):
        P = expert_matrix(sd, experts, i)
        mix = w @ P
        y = int(sd.y[i])
        pa = P[:, y]
        losses -= np.log(np.maximum(pa, 1e-300))
        rep.mix_loss -= math.log(max(float(mix[y]), 1e-300))
        nw = w * np.power(np.maximum(pa, 1e-300), eta)
        nw /= nw.sum()
        nw = (1 - alpha) * nw + alpha / k
        order = np.argsort(-P, axis=1, kind="stable")
        ranks = np.argmax(order == y, axis=1) + 1
        rep.steps.append(Step(sd.dates[i], y, mix, w, nw, pa, ranks, P.mean(axis=0)))
        w = nw
    rep.weights = w
    rep.losses = losses
    return rep


def predict_next(sd: SeriesData, rep: Replay, date: dt.date) -> tuple[np.ndarray, np.ndarray]:
    """(final distribution, expert matrix) for the next unknown draw on `date`."""
    P = expert_matrix(sd, rep.experts, sd.n, date)
    return rep.weights @ P, P


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
