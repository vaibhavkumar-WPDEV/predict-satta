"""The experts. Each one is a different mathematical theory of where the next
number comes from, and each returns P(next jodi = v) for v = 00..99.

The ensemble (ensemble.py) decides every day, from real results, how much to
trust each expert. An expert that does not beat chance loses weight.
"""

from __future__ import annotations

import numpy as np

from . import formulas
from .base import UNIFORM, UNIFORM10, Context, cut, normalize, outer, rev


class Expert:
    name = "expert"
    label = ""
    theory = ""

    def predict(self, ctx: Context) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class Uniform(Expert):
    name = "uniform"
    label = "Random baseline"
    theory = "P(v) = 1/100. Agar numbers sach me random hain to koi bhi model isse behtar nahi ho sakta."

    def predict(self, ctx):
        return UNIFORM.copy()


class Frequency(Expert):
    """Dirichlet-multinomial posterior predictive."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.name = f"freq_a{alpha:g}"
        self.label = f"Frequency (Bayes, alpha={alpha:g})"
        self.theory = "P(v) = (count(v) + a) / (n + 100a)  —  Dirichlet prior ke saath kitni baar aaya."

    def predict(self, ctx):
        c = np.bincount(ctx.y, minlength=100).astype(float) + self.alpha
        return c / c.sum()


class Recency(Expert):
    """Exponentially decayed counts: recent draws count more."""

    def __init__(self, half_life: float):
        self.h = half_life
        self.name = f"recency_h{half_life:g}"
        self.label = f"Recency (half-life {half_life:g} din)"
        self.theory = "P(v) ∝ Σ 2^(-age/h)·[y=v] + 0.5  —  naye results ka wazan zyada."

    def predict(self, ctx):
        n = len(ctx.y)
        if n == 0:
            return UNIFORM.copy()
        w = 0.5 ** (np.arange(n)[::-1] / self.h)
        c = np.bincount(ctx.y, weights=w, minlength=100) + 0.5
        return c / c.sum()


class RecencyDigit(Expert):
    """Hot Andar / Bahar digits with exponential decay."""

    def __init__(self, half_life: float):
        self.h = half_life
        self.name = f"hot_digit_h{half_life:g}"
        self.label = f"Hot Andar-Bahar (half-life {half_life:g})"
        self.theory = "P(v) = P(andar)·P(bahar), har digit ki decayed frequency se."

    def predict(self, ctx):
        n = len(ctx.y)
        if n == 0:
            return UNIFORM.copy()
        w = 0.5 ** (np.arange(n)[::-1] / self.h)
        pt = np.bincount(ctx.y // 10, weights=w, minlength=10) + 0.5
        pu = np.bincount(ctx.y % 10, weights=w, minlength=10) + 0.5
        return outer(pt, pu)


def _digit_markov(y: np.ndarray, cross: bool) -> np.ndarray:
    if len(y) < 2:
        return UNIFORM.copy()
    t, u = y // 10, y % 10
    src_t, src_u = (u, t) if cross else (t, u)
    T = np.ones((10, 10))
    U = np.ones((10, 10))
    np.add.at(T, (src_t[:-1], t[1:]), 1)
    np.add.at(U, (src_u[:-1], u[1:]), 1)
    return outer(T[src_t[-1]], U[src_u[-1]])


class MarkovDigit(Expert):
    def __init__(self, cross: bool):
        self.cross = cross
        self.name = "markov_digit_x" if cross else "markov_digit"
        self.label = "Markov digit (palti-cross)" if cross else "Markov chain (Andar→Andar, Bahar→Bahar)"
        self.theory = ("P(andar_t | bahar_(t-1))·P(bahar_t | andar_(t-1))" if cross
                       else "P(andar_t | andar_(t-1))·P(bahar_t | bahar_(t-1)), Laplace smoothing.")

    def predict(self, ctx):
        return _digit_markov(ctx.y, self.cross)


class MarkovJodi(Expert):
    name = "markov_jodi"
    label = "Markov chain (jodi→jodi)"
    theory = "P(v | pichla=x) = (C[x→v] + k·P_digit(v)) / (C[x] + k)  —  sparse hone par digit model par back-off."

    def predict(self, ctx):
        y = ctx.y
        if len(y) < 3:
            return UNIFORM.copy()
        nxt = y[1:][y[:-1] == y[-1]]
        base = _digit_markov(y, False)
        k = 3.0
        return (np.bincount(nxt, minlength=100) + k * base) / (len(nxt) + k)


class GapHazard(Expert):
    """Survival analysis: probability a number shows given how long it has been absent."""

    name = "gap_hazard"
    label = "Gap / overdue hazard"
    theory = ("Piecewise-constant hazard h(g) = events_g / exposure_g (Kaplan–Meier style). "
              "P(v) ∝ h(gap_v). Random game me h(g) har g par 1% hoga.")
    EDGES = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 10 ** 9])

    def predict(self, ctx):
        y = ctx.y
        n = len(y)
        if n < 30:
            return UNIFORM.copy()
        lo = self.EDGES[:-1]
        hi = self.EDGES[1:] - 1
        events = np.zeros(len(lo))
        exposure = np.zeros(len(lo))
        last = np.full(100, -1)
        gaps = []
        for i, v in enumerate(y):
            if last[v] >= 0:
                gaps.append(i - last[v])
            last[v] = i
        if gaps:
            G = np.array(gaps)
            b = np.searchsorted(self.EDGES, G, side="right") - 1
            np.add.at(events, b, 1)
            exposure += np.clip(np.minimum(G[:, None], hi[None, :]) - lo[None, :] + 1, 0, None).sum(axis=0)
        a = 100.0
        hazard = (events + a * 0.01) / (exposure + a)
        cur_gap = np.where(last >= 0, n - last, n + 1)
        b = np.searchsorted(self.EDGES, cur_gap, side="right") - 1
        return normalize(hazard[np.clip(b, 0, len(hazard) - 1)])


class CrossMarket(Expert):
    name = "cross_market"
    label = "Cross-market (kal ke doosre market)"
    theory = "P(andar | andar of M kal)·P(bahar | bahar of M kal), sab markets M ka average."

    def predict(self, ctx):
        y = ctx.y
        pts, pus = [], []
        for m in ctx.sd.others:
            cur = ctx.cur[m]
            if cur < 0:
                continue
            x = ctx.past(m)
            ok = x >= 0
            if ok.sum() < 20:
                continue
            xt, xu, yt, yu = x[ok] // 10, x[ok] % 10, y[ok] // 10, y[ok] % 10
            pts.append(np.bincount(yt[xt == cur // 10], minlength=10) + 1.0)
            pus.append(np.bincount(yu[xu == cur % 10], minlength=10) + 1.0)
        if not pts:
            return UNIFORM.copy()
        pt = np.mean([normalize(p) for p in pts], axis=0)
        pu = np.mean([normalize(p) for p in pus], axis=0)
        return outer(pt, pu)


class Weekday(Expert):
    name = "weekday"
    label = "Hafte ka din pattern"
    theory = "P(andar | weekday)·P(bahar | weekday), Laplace smoothing."

    def predict(self, ctx):
        wd = ctx.past("WD")
        mask = wd == ctx.cur["WD"]
        if mask.sum() < 5:
            return UNIFORM.copy()
        yy = ctx.y[mask]
        return outer(np.bincount(yy // 10, minlength=10) + 2.0, np.bincount(yy % 10, minlength=10) + 2.0)


TRANSFORMS = {
    "same": lambda x: x,
    "palti": rev,
    "cut": cut,
    "palti+cut": lambda x: cut(rev(x)),
    "+1": lambda x: (x + 1) % 100,
    "-1": lambda x: (x - 1) % 100,
    "+10": lambda x: (x + 10) % 100,
    "-10": lambda x: (x - 10) % 100,
    "+11": lambda x: (x + 11) % 100,
    "-11": lambda x: (x - 11) % 100,
    "99-x": lambda x: 99 - x,
    "jod-jodi": lambda x: ((x // 10 + x % 10) % 10) * 11,
}


class Transforms(Expert):
    name = "transforms"
    label = "Satta tricks (palti, cut, ±1, ±11 ...)"
    theory = ("Har rule r = T(source) ke liye Beta posterior hit-rate q_r; weight ∝ exp(z_r). "
              "P = Σ w_r [q_r·δ(T(src)) + (1-q_r)/100].")

    def predict(self, ctx):
        y = ctx.y
        if len(y) < 20:
            return UNIFORM.copy()
        srcs = {"A1": ctx.past("A1")}
        for m in ctx.sd.others:
            srcs[m] = ctx.past(m)
        num = np.zeros(100)
        wsum = 0.0
        for s, x in srcs.items():
            cur = ctx.cur[s]
            if cur < 0:
                continue
            ok = x >= 0
            trials = int(ok.sum())
            if trials < 15:
                continue
            for fn in TRANSFORMS.values():
                hits = int((fn(x[ok]) == y[ok]).sum())
                q = (hits + 0.2) / (trials + 20.0)
                z = (hits - 0.01 * trials) / np.sqrt(trials * 0.0099)
                w = float(np.exp(np.clip(z, -3, 3)))
                dist = np.full(100, (1 - q) / 100)
                dist[int(fn(cur))] += q
                num += w * dist
                wsum += w
        return normalize(num) if wsum else UNIFORM.copy()


class Analog(Expert):
    """k-nearest-neighbour on the last k draws (method of analogues)."""

    def __init__(self, k: int):
        self.k = k
        self.name = f"analog_k{k}"
        self.label = f"Pattern match (pichle {k} din jaisa itihaas)"
        self.theory = ("sim_j = Σ_l 2[y_(j-l)=y_(n-l)] + [andar same] + [bahar same]; "
                       "P ∝ Σ_j exp(0.7·sim_j)·δ(y_j) (andar/bahar par).")

    def predict(self, ctx):
        y = ctx.y
        n, k = len(y), self.k
        if n < k + 10:
            return UNIFORM.copy()
        t, u = y // 10, y % 10
        J = np.arange(k, n)
        sim = np.zeros(len(J))
        for l in range(1, k + 1):
            sim += 2 * (y[J - l] == y[n - l]) + (t[J - l] == t[n - l]) + (u[J - l] == u[n - l])
        w = np.exp(0.7 * (sim - sim.max()))
        pt = np.bincount(t[J], weights=w, minlength=10) + 0.05 * w.sum()
        pu = np.bincount(u[J], weights=w, minlength=10) + 0.05 * w.sum()
        return outer(pt, pu)


def _harmonic_forecast(b: np.ndarray, top: int = 3, max_period: int = 40) -> float:
    """Pick the strongest periods from a periodogram, fit them by least squares, extrapolate 1 step."""
    N = len(b)
    periods = np.arange(2, min(max_period, N // 3) + 1)
    if len(periods) == 0:
        return float(b.mean())
    t = np.arange(N)
    bm = b - b.mean()
    W = 2 * np.pi / periods[:, None] * t[None, :]
    power = (np.cos(W) @ bm) ** 2 + (np.sin(W) @ bm) ** 2
    best = periods[np.argsort(-power)[:top]]
    cols = [np.ones(N)]
    nxt = [1.0]
    for p in best:
        cols += [np.cos(2 * np.pi * t / p), np.sin(2 * np.pi * t / p)]
        nxt += [np.cos(2 * np.pi * N / p), np.sin(2 * np.pi * N / p)]
    X = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(X, b, rcond=None)
    return float(np.dot(nxt, coef))


class Spectral(Expert):
    name = "spectral"
    label = "Fourier / spectral cycles"
    theory = ("Har digit d ke liye indicator series b_t=[digit_t=d] ka periodogram; top-3 periods ka "
              "least-squares harmonic fit aur agle din tak extrapolation.")
    WINDOW = 180

    def predict(self, ctx):
        y = ctx.y[-self.WINDOW:]
        if len(y) < 30:
            return UNIFORM.copy()
        out = []
        for digits in (y // 10, y % 10):
            score = np.array([_harmonic_forecast((digits == d).astype(float)) for d in range(10)])
            freq = normalize(np.bincount(digits, minlength=10) + 1.0)
            out.append(0.5 * normalize(np.clip(score, 0.005, None)) + 0.5 * freq)
        return outer(*out)


class FormulaJodi(Expert):
    name = "formula_jodi"
    label = "Khud ke formule (jodi)"
    theory = ("Har 7 din par ~250 formule (c1·U + c2·V + k) mod 100 dobara dhoondhe jaate hain; "
              "top-5 vote karte hain: P = 0.75/100 + 0.25·Σ w_f δ(f).")
    REFRESH = 7
    TOP = 5

    def predict(self, ctx):
        i = ctx.i
        r = (i // self.REFRESH) * self.REFRESH
        if r < 30:
            return UNIFORM.copy()
        key = ("fjodi", r)
        if key not in ctx.sd.cache:
            ctx.sd.cache[key] = formulas.discover(ctx.sd, r, "jodi", self.TOP)
        found = ctx.sd.cache[key]
        jv, _ = formulas.current_vars(ctx.sd, ctx.cur)
        dist = np.zeros(100)
        for f in found:
            v = int(formulas.apply(f, jv)[0])
            if v >= 0:
                dist[v] += max(f["hits"] / f["n"] - 0.01, 1e-3)
        if dist.sum() == 0:
            return UNIFORM.copy()
        return 0.75 * UNIFORM + 0.25 * normalize(dist)


class FormulaDigit(Expert):
    name = "formula_digit"
    label = "Khud ke formule (andar/bahar)"
    theory = ("Andar aur Bahar ke liye alag (c1·u + c2·v + k) mod 10 formule, har 7 din par "
              "dobara search; top-3 ka vote, 60% uniform ke saath.")
    REFRESH = 7
    TOP = 3

    def predict(self, ctx):
        i = ctx.i
        r = (i // self.REFRESH) * self.REFRESH
        if r < 30:
            return UNIFORM.copy()
        _, dv = formulas.current_vars(ctx.sd, ctx.cur)
        parts = []
        for kind in ("andar", "bahar"):
            key = ("f" + kind, r)
            if key not in ctx.sd.cache:
                ctx.sd.cache[key] = formulas.discover(ctx.sd, r, kind, self.TOP)
            dist = np.zeros(10)
            for f in ctx.sd.cache[key]:
                v = int(formulas.apply(f, dv)[0])
                if v >= 0:
                    dist[v] += max(f["hits"] / f["n"] - 0.1, 1e-3)
            parts.append(0.6 * UNIFORM10 + 0.4 * normalize(dist) if dist.sum() else UNIFORM10)
        return outer(*parts)


def default_experts() -> list[Expert]:
    return [
        Uniform(),
        Frequency(1.0),
        Frequency(20.0),
        Recency(7),
        Recency(30),
        Recency(90),
        RecencyDigit(5),
        RecencyDigit(20),
        MarkovJodi(),
        MarkovDigit(False),
        MarkovDigit(True),
        GapHazard(),
        CrossMarket(),
        Weekday(),
        Transforms(),
        Analog(2),
        Analog(3),
        Spectral(),
        FormulaJodi(),
        FormulaDigit(),
    ]
