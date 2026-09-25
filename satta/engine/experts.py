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
        self.label = f"Bayes–Laplace frequency (alpha={alpha:g})"
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
        self.label = "Markov chain (palti-cross digits)" if cross else "Markov chain (Andar→Andar, Bahar→Bahar)"
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
    label = "Survival analysis (Kaplan–Meier gap hazard)"
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
        rules = []
        for s, x in srcs.items():
            cur = ctx.cur[s]
            if cur < 0:
                continue
            for fn in TRANSFORMS.values():
                rules.append((np.where(x >= 0, fn(np.where(x >= 0, x, 0)), -1), int(fn(cur))))
        return rule_mixture(rules, y)


def rule_mixture(rules: list[tuple[np.ndarray, int]], y: np.ndarray, min_trials: int = 15) -> np.ndarray:
    """Mix deterministic rules "next = f(past)" by how often each was right.

    rules: (predictions for every past draw, -1 = not available; prediction for the next draw).
    Each rule gets a Beta-posterior hit rate q and weight exp(z), z = excess hits in sigmas.
    """
    num = np.zeros(100)
    wsum = 0.0
    for past, cur in rules:
        if cur < 0:
            continue
        ok = past >= 0
        trials = int(ok.sum())
        if trials < min_trials:
            continue
        hits = int((past[ok] == y[ok]).sum())
        q = (hits + 0.2) / (trials + 20.0)
        z = (hits - 0.01 * trials) / np.sqrt(trials * 0.0099)
        w = float(np.exp(np.clip(z, -3, 3)))
        dist = np.full(100, (1 - q) / 100)
        dist[cur % 100] += q
        num += w * dist
        wsum += w
    return normalize(num) if wsum else UNIFORM.copy()


def confident_mix(votes: np.ndarray, best_rate: float, base: np.ndarray) -> np.ndarray:
    """Trust the formula votes as much as the best formula was right in the past (5%..95%).

    On random data the best of thousands of formulas is right only a few % of the time, so the
    output stays close to uniform; on data that really follows a formula it becomes sharp.
    """
    if votes.sum() == 0:
        return base.copy()
    g = float(np.clip(best_rate, 0.05, 0.95))
    return (1 - g) * base + g * normalize(votes)


def _lagged(y: np.ndarray, lag: int) -> np.ndarray:
    """out[t] = y[t-lag] (or -1), aligned with y."""
    out = np.full(len(y), -1)
    if lag < len(y):
        out[lag:] = y[:-lag]
    return out


PHI = (1 + 5 ** 0.5) / 2
FIB_LAGS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]


class Fibonacci(Expert):
    name = "fibonacci_golden"
    label = "Fibonacci / Golden ratio (Pingala, Da Vinci)"
    theory = ("Rules: y_t = (y_(t-1) + y_(t-2)) mod 100; andar/bahar Fibonacci; golden rotation "
              "y_t = (y_(t-1) + 100/φ) mod 100 aur +100/φ²; Fibonacci lag echo y_t = y_(t-F), "
              "F ∈ {1,2,3,5,8,13,21,34,55,89}. Har rule ka asli hit-rate uska weight tay karta hai.")

    def predict(self, ctx):
        y = ctx.y
        n = len(y)
        if n < 25:
            return UNIFORM.copy()
        a1, a2 = _lagged(y, 1), _lagged(y, 2)
        ok = (a1 >= 0) & (a2 >= 0)
        rules = [
            (np.where(ok, (a1 + a2) % 100, -1), int((y[-1] + y[-2]) % 100)),
            (np.where(ok, ((a1 // 10 + a2 // 10) % 10) * 10 + (a1 + a2) % 10, -1),
             int(((y[-1] // 10 + y[-2] // 10) % 10) * 10 + (y[-1] + y[-2]) % 10)),
        ]
        for shift in (100 / PHI, 100 / PHI ** 2):
            s = int(round(shift))
            rules.append((np.where(a1 >= 0, (a1 + s) % 100, -1), int((y[-1] + s) % 100)))
        for lag in FIB_LAGS:
            if lag < n:
                rules.append((_lagged(y, lag), int(y[n - lag])))
        return rule_mixture(rules, y)


class Aryabhata(Expert):
    """Linear congruence solver (kuttaka): the rule that would crack a linear congruential generator."""

    name = "aryabhata_lcg"
    label = "Aryabhata kuttaka (a·pichla + c) mod 100"
    theory = ("Sab 100×100 linear congruences y_t ≡ a·y_(t-1) + c (mod 100) try; har a ke liye c = sabse "
              "zyada aane wala residual. Top-3 (a, c) vote karte hain (bharosa = unka hit-rate), har 7 din dobara solve. "
              "Agar numbers kisi LCG random generator se bante, yeh use pakad leta.")
    REFRESH = 7

    def predict(self, ctx):
        r = (ctx.i // self.REFRESH) * self.REFRESH
        if r < 30:
            return UNIFORM.copy()
        key = ("lcg", r)
        if key not in ctx.sd.cache:
            ctx.sd.cache[key] = formulas.lcg_solve(ctx.sd.y[:r])
        dist = np.zeros(100)
        for a, c, hits, n in ctx.sd.cache[key]:
            dist[(a * int(ctx.y[-1]) + c) % 100] += max(hits / n - 0.01, 1e-3)
        found = ctx.sd.cache[key]
        return confident_mix(dist, found[0][2] / found[0][3], UNIFORM)


class AryabhataDigit(Expert):
    name = "aryabhata_digit"
    label = "Aryabhata kuttaka (andar/bahar, 2nd order)"
    theory = ("Har digit ke liye d_t ≡ a·d_(t-1) + b·d_(t-2) + c (mod 10), sab a, b ∈ 0..9 try; "
              "top-3 vote; bharosa g = best formula ka purana hit-rate (5–95%).")
    REFRESH = 7

    @staticmethod
    def solve(d: np.ndarray, top: int = 3):
        d1, d2, t = d[1:-1], d[:-2], d[2:]
        a = np.arange(10)[:, None, None]
        b = np.arange(10)[None, :, None]
        R = (t[None, None, :] - a * d1[None, None, :] - b * d2[None, None, :]) % 10
        idx = ((a * 10 + b) * 10 + R).ravel()
        counts = np.bincount(idx, minlength=1000)
        best = np.argsort(-counts, kind="stable")[:top]
        return [(int(k // 100), int(k // 10 % 10), int(k % 10), int(counts[k]), len(t)) for k in best]

    def predict(self, ctx):
        r = (ctx.i // self.REFRESH) * self.REFRESH
        if r < 30:
            return UNIFORM.copy()
        parts = []
        for name, digits in (("t", ctx.sd.y // 10), ("u", ctx.sd.y % 10)):
            key = ("lcg_digit", name, r)
            if key not in ctx.sd.cache:
                ctx.sd.cache[key] = self.solve(digits[:r])
            cur = digits[: ctx.i]
            dist = np.zeros(10)
            for a, b, c, hits, n in ctx.sd.cache[key]:
                dist[(a * int(cur[-1]) + b * int(cur[-2]) + c) % 10] += max(hits / n - 0.1, 1e-3)
            best = ctx.sd.cache[key][0]
            parts.append(confident_mix(dist, best[3] / best[4], UNIFORM10))
        return outer(*parts)


def digital_root(x):
    """Vedic beejank: 0 -> 0, otherwise 1 + (x - 1) mod 9."""
    x = np.asarray(x)
    return np.where(x == 0, 0, 1 + (x - 1) % 9)


DR_ALL = digital_root(np.arange(100))
DR_SIZE = np.bincount(DR_ALL, minlength=10)


class VedicTesla(Expert):
    name = "vedic_tesla"
    label = "Vedic beejank / Tesla 3-6-9 (digital root)"
    theory = ("Beejank dr(x) = 1 + (x−1) mod 9. Markov chain P(dr_t | dr_(t-1)) (3-6-9 sameti); "
              "P(v) = P(dr(v)) / |{x : dr(x) = dr(v)}|.")

    def predict(self, ctx):
        y = ctx.y
        if len(y) < 30:
            return UNIFORM.copy()
        dr = digital_root(y)
        T = np.ones((10, 10))
        np.add.at(T, (dr[:-1], dr[1:]), 1)
        p_dr = normalize(T[dr[-1]])
        return normalize(p_dr[DR_ALL] / DR_SIZE[DR_ALL])


class RandomWalk(Expert):
    name = "einstein_walk"
    label = "Einstein Brownian motion (random walk)"
    theory = ("Badlaav Δ = (y_t − y_(t-1)) mod 100 ka distribution, circular Gaussian kernel (σ=1.5) "
              "se smooth: P(v) = P_Δ((v − pichla) mod 100). Diffusion model.")
    KERNEL = np.exp(-0.5 * (np.minimum(np.arange(100), 100 - np.arange(100)) / 1.5) ** 2)

    def predict(self, ctx):
        y = ctx.y
        if len(y) < 30:
            return UNIFORM.copy()
        delta = np.bincount((y[1:] - y[:-1]) % 100, minlength=100).astype(float)
        smooth = np.real(np.fft.ifft(np.fft.fft(delta) * np.fft.fft(self.KERNEL / self.KERNEL.sum())))
        p_delta = normalize(np.clip(smooth, 0, None) + 0.5)
        return normalize(np.roll(p_delta, int(y[-1])))


class Analog(Expert):
    """k-nearest-neighbour on the last k draws (method of analogues)."""

    def __init__(self, k: int):
        self.k = k
        self.name = f"analog_k{k}"
        self.label = f"Chaos theory: Takens embedding (pichle {k} din jaisa itihaas)"
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


_BASIS: dict[int, tuple] = {}


def _basis(N: int, max_period: int = 40):
    """cos/sin rows for periods 2..max_period over t = 0..N-1, plus their values at t = N."""
    if N not in _BASIS:
        periods = np.arange(2, min(max_period, N // 3) + 1)
        W = 2 * np.pi / periods[:, None] * np.arange(N + 1)[None, :]
        _BASIS[N] = (np.cos(W[:, :N]), np.sin(W[:, :N]), np.cos(W[:, N]), np.sin(W[:, N]))
    return _BASIS[N]


def _harmonic_forecast(B: np.ndarray, top: int = 3) -> np.ndarray:
    """For each row of B (one 0/1 series per digit): pick the strongest periods from the
    periodogram, fit them by least squares and extrapolate one step ahead."""
    N = B.shape[1]
    C, S, cN, sN = _basis(N)
    if len(C) == 0:
        return B.mean(axis=1)
    Bm = B - B.mean(axis=1, keepdims=True)
    power = (C @ Bm.T) ** 2 + (S @ Bm.T) ** 2          # (periods, rows)
    out = np.empty(len(B))
    for r in range(len(B)):
        best = np.argsort(-power[:, r])[:top]
        X = np.column_stack([np.ones(N), C[best].T, S[best].T])
        coef, *_ = np.linalg.lstsq(X, B[r], rcond=None)
        out[r] = np.concatenate([[1.0], cN[best], sN[best]]) @ coef
    return out


class Spectral(Expert):
    name = "spectral"
    label = "Fourier cycles (spectral analysis)"
    theory = ("Har digit d ke liye indicator series b_t=[digit_t=d] ka periodogram; top-3 periods ka "
              "least-squares harmonic fit aur agle din tak extrapolation.")
    WINDOW = 180

    def predict(self, ctx):
        y = ctx.y[-self.WINDOW:]
        if len(y) < 30:
            return UNIFORM.copy()
        out = []
        for digits in (y // 10, y % 10):
            score = _harmonic_forecast((digits[None, :] == np.arange(10)[:, None]).astype(float))
            freq = normalize(np.bincount(digits, minlength=10) + 1.0)
            out.append(0.5 * normalize(np.clip(score, 0.005, None)) + 0.5 * freq)
        return outer(*out)


class FormulaJodi(Expert):
    name = "formula_jodi"
    label = "Khud ke formule (jodi)"
    theory = ("Har 7 din par ~250 formule (c1·U + c2·V + k) mod 100 dobara dhoondhe jaate hain; "
              "top-5 vote karte hain: P = (1−g)/100 + g·Σ w_f δ(f), g = best formula ka hit-rate (5–95%).")
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
        return confident_mix(dist, found[0]["hits"] / found[0]["n"], UNIFORM)


class FormulaDigit(Expert):
    name = "formula_digit"
    label = "Khud ke formule (andar/bahar)"
    theory = ("Andar aur Bahar ke liye alag (c1·u + c2·v + k) mod 10 formule, har 7 din par "
              "dobara search; top-3 ka vote, bharosa g = best formula ka hit-rate.")
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
            found = ctx.sd.cache[key]
            parts.append(confident_mix(dist, found[0]["hits"] / found[0]["n"], UNIFORM10)
                         if dist.sum() else UNIFORM10)
        return outer(*parts)


def default_experts() -> list[Expert]:
    from .advanced import NeuralNet, UniversalCTW

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
        Fibonacci(),
        Aryabhata(),
        AryabhataDigit(),
        VedicTesla(),
        RandomWalk(),
        UniversalCTW(),
        NeuralNet(),
    ]
