"""Advanced learners: universal prediction (CTW) and an online neural network.

Both keep their trained state in sd.cache and only ever train on draws before
the one being predicted, so walk-forward results stay leak-free. Training is
deterministic (fixed seeds per sample), so the result does not depend on how
the replay is chunked.
"""

from __future__ import annotations

import math

import numpy as np

from .base import UNIFORM, Context, outer
from .experts import Expert

LOG_HALF = math.log(0.5)


class CTW:
    """Context Tree Weighting (Willems, Shtarkov, Tjalkens 1995) for a 10-letter alphabet.

    Every context of length 0..D gets a Krichevsky–Trofimov estimator; a node's
    weighted probability is ½·P_e(node) + ½·Π P_w(children). The root therefore
    averages all tree models (all Markov orders <= D, with pruning) in closed
    form. Redundancy vs the best tree source is O(log n) — a universal predictor.
    """

    def __init__(self, depth: int = 3, m: int = 10):
        self.D = depth
        self.m = m
        self.nodes: dict[tuple, list] = {}  # ctx -> [counts, total, logPe, logPw, sum_child_logPw]
        self.hist: list[int] = []

    def _node(self, ctx):
        nd = self.nodes.get(ctx)
        if nd is None:
            nd = [np.zeros(self.m), 0, 0.0, 0.0, 0.0]
            self.nodes[ctx] = nd
        return nd

    def _path(self):
        h = self.hist
        return [tuple(h[-1:-d - 1:-1]) if d else () for d in range(self.D + 1)]

    def _new_values(self, x: int):
        """logPw of every node on the current path if x were the next symbol (leaf first)."""
        path = self._path()
        out = []
        child_old = child_new = 0.0
        for d in range(self.D, -1, -1):
            nd = self.nodes.get(path[d])
            counts, total, lpe, lpw, sch = nd if nd is not None else (None, 0, 0.0, 0.0, 0.0)
            cx = counts[x] if counts is not None else 0.0
            lpe_new = lpe + math.log((cx + 0.5) / (total + 0.5 * self.m))
            if d == self.D:
                lpw_new = lpe_new
                sch_new = sch
            else:
                sch_new = sch - child_old + child_new
                lpw_new = np.logaddexp(LOG_HALF + lpe_new, LOG_HALF + sch_new)
            out.append((path[d], lpe_new, lpw_new, sch_new, lpw))
            child_old, child_new = lpw, lpw_new
        return out

    def predict(self) -> np.ndarray:
        if len(self.hist) < self.D:
            return np.full(self.m, 1.0 / self.m)
        root = self.nodes.get(())
        root_lpw = root[3] if root else 0.0
        lp = np.array([self._new_values(x)[-1][2] - root_lpw for x in range(self.m)])
        p = np.exp(lp - lp.max())
        return p / p.sum()

    def update(self, x: int) -> None:
        if len(self.hist) >= self.D:
            for ctx, lpe, lpw, sch, _ in self._new_values(x):
                nd = self._node(ctx)
                nd[0][x] += 1
                nd[1] += 1
                nd[2], nd[3], nd[4] = lpe, lpw, sch
        self.hist.append(x)


class UniversalCTW(Expert):
    name = "universal_ctw"
    label = "Universal prediction: Context Tree Weighting"
    theory = ("Andar aur Bahar dono ke liye depth-3 context tree: har context (0-3 pichle digits) ka "
              "KT estimator, P_w = ½P_e + ½ΠP_w(children). Saare Markov orders ka exact Bayesian "
              "average (Willems 1995) — kisi bhi finite-memory pattern ko optimal speed se seekhta hai.")
    DEPTH = 3

    def predict(self, ctx: Context):
        key = "ctw"
        state = ctx.sd.cache.get(key)
        if state is None or state[0] > ctx.i:
            state = [0, CTW(self.DEPTH), CTW(self.DEPTH)]
            ctx.sd.cache[key] = state
        y = ctx.sd.y
        while state[0] < ctx.i:
            v = int(y[state[0]])
            state[1].update(v // 10)
            state[2].update(v % 10)
            state[0] += 1
        return outer(state[1].predict(), state[2].predict())


def _onehot_features(sd, F: dict) -> np.ndarray:
    """Binary features known before the draw: digits of the last two draws, other markets'
    previous-day digits, weekday, plus a bias. Missing values give all-zero blocks."""
    cols = []
    for key in ["A1", "A2"] + sd.cross_keys:
        x = np.asarray(F[key])
        for digit in (x // 10, x % 10):
            oh = (digit[:, None] == np.arange(10)[None, :]) & (x >= 0)[:, None]
            cols.append(oh.astype(float))
    wd = np.asarray(F["WD"])
    cols.append((wd[:, None] == np.arange(7)[None, :]).astype(float))
    # cyclical time: sin/cos keep 31 → 1, Dec → Jan, Sun → Mon next to each other
    dm, mo = np.asarray(F["DM"], float), np.asarray(F["MO"], float)
    for value, period in ((dm, 31.0), (mo, 12.0), (wd.astype(float), 7.0)):
        cols.append(np.stack([np.sin(2 * np.pi * value / period), np.cos(2 * np.pi * value / period)], axis=1))
    cols.append(np.ones((len(wd), 1)))
    return np.concatenate(cols, axis=1)


class Symbolic(Expert):
    name = "symbolic"
    label = "Numerology + Chandra tithi (symbolic hypothesis)"
    theory = ("Tareekh, mahina, saal, mulank, bhagyank aur chandra tithi se bane 14 rules (jaise "
              "'mulank x2', 'DD+MM', 'tithi'). Har rule ka asli purana hit-rate uska weight tay karta "
              "hai. Vaigyanik saboot nahi — sirf data par test kiya jaane wala hypothesis.")

    def predict(self, ctx: Context):
        from . import symbolic
        from .experts import rule_mixture

        sd = ctx.sd
        if "sym" not in sd.cache:
            sd.cache["sym"] = symbolic.rule_table(sd.dates)
        R = sd.cache["sym"]
        if ctx.i < 60:
            return UNIFORM.copy()
        cur = symbolic.candidates(ctx.date)
        rules = [(R[: ctx.i, j], cur[name]) for j, name in enumerate(symbolic.RULES)]
        return rule_mixture(rules, ctx.y)


class HotPool(Expert):
    """Cross-market recency: numbers that came recently in ANY market come again a little
    more often than chance (found on 2021-2026 data: Disawar 12.3 % top-10 vs 10 %)."""

    GRID = np.array([0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.3])

    def __init__(self, window: int, decay: float):
        self.W = window
        self.decay = decay
        self.name = f"hot_pool_w{window}"
        self.label = f"Hot pool: number ghoom ke aata hai ({window} din, sab markets)"
        self.theory = (f"score(v) = Σ {decay}^age · [v aaya] pichle {window} din me chaaron markets me "
                       "(aaj ke pehle aaye markets bhi). P = (1−λ)/100 + λ·score/Σscore; λ har din "
                       "pichle 730 draws par maximum likelihood se khud chuna jata hai.")

    def _scores(self, sd, date) -> np.ndarray:
        import datetime as dt

        from .. import config

        sc = np.zeros(100)
        for k in range(self.W + 1):
            day = date - dt.timedelta(days=k)
            w = self.decay ** k
            for o in config.MARKET_KEYS:
                if k == 0 and o not in sd.earlier:
                    continue  # same day: only markets declared before this one
                v = sd.table.get(o, {}).get(day)
                if v is not None:
                    sc[v] += w
        return sc

    def predict(self, ctx: Context):
        sd = ctx.sd
        q = sd.cache.setdefault(("hot", self.W, self.decay), [])
        while len(q) < ctx.i:  # pool probability the real number had, for every past draw
            j = len(q)
            sc = self._scores(sd, sd.dates[j])
            s = sc.sum()
            q.append(sc[sd.y[j]] / s if s > 0 else 0.01)
        sc = self._scores(sd, ctx.date)
        s = sc.sum()
        if s == 0 or ctx.i < 60:
            return UNIFORM.copy()
        qa = np.array(q[max(0, ctx.i - 730):ctx.i])
        ll = [np.log((1 - lam) / 100 + lam * qa).sum() for lam in self.GRID]
        lam = float(self.GRID[int(np.argmax(ll))])
        return (1 - lam) * UNIFORM + lam * sc / s


class GeneticFormula(Expert):
    name = "genetic_formula"
    label = "Genetic programming (khud evolve kiye formule)"
    theory = ("Har 28 draws par 60 random formule (+ − ×, ulta, cut, andar/bahar, jod, beejank, jodi) "
              "12 generations tak evolve (selection, crossover, mutation) pichle 730 draws par; top-3 "
              "vote karte hain, bharosa = best formula ka purana hit-rate.")
    REFRESH = 28

    def predict(self, ctx: Context):
        from . import formulas, genetic
        from .experts import confident_mix

        r = (ctx.i // self.REFRESH) * self.REFRESH
        if r < 60:
            return UNIFORM.copy()
        key = ("gp", r)
        if key not in ctx.sd.cache:
            jv, _, _ = formulas._series_vars(ctx.sd)
            lo = max(0, r - formulas.WINDOW)
            ctx.sd.cache[key] = genetic.evolve({k: v[lo:r] for k, v in jv.items()}, ctx.sd.y[lo:r],
                                               seed=r, pop=60, gens=12, top=3)
        found = ctx.sd.cache[key]
        if not found:
            return UNIFORM.copy()
        cjv, _ = formulas.current_vars(ctx.sd, ctx.cur)
        votes = np.zeros(100)
        for f in found:
            v = genetic.predict(f, cjv)
            if v >= 0:
                votes[v] += max(f["hits"] / f["n"] - 0.01, 1e-3)
        return confident_mix(votes, found[0]["hits"] / found[0]["n"], UNIFORM)


class NeuralNet(Expert):
    """One-hidden-layer network trained online by SGD with experience replay."""

    name = "neural_net"
    label = "Neural network (online, 32 neurons)"
    theory = ("Input: pichle 2 draws + doosre markets ke kal/aaj ke andar/bahar (one-hot) + weekday + "
              "tareekh/mahina/din ke sin-cos (cyclical time). "
              "Hidden: 32 tanh neurons. Output: do softmax (andar, bahar). Har naye result par SGD + "
              "4 purane samples ka replay, L2 regularisation. P = 0.7·outer(andar, bahar) + 0.3·uniform.")
    H = 32
    LR = 0.03
    L2 = 1e-4
    REPLAY = 4

    def _data(self, sd):
        if "nn_X" not in sd.cache:
            sd.cache["nn_X"] = _onehot_features(sd, sd.F) if sd.n else np.zeros((0, 1))
        return sd.cache["nn_X"]

    def _init(self, d):
        rng = np.random.default_rng(7)
        return {"W1": rng.normal(0, 0.1, (d, self.H)), "b1": np.zeros(self.H),
                "Wa": rng.normal(0, 0.1, (self.H, 10)), "ba": np.zeros(10),
                "Wb": rng.normal(0, 0.1, (self.H, 10)), "bb": np.zeros(10)}

    @staticmethod
    def _softmax(z):
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def _forward(self, p, x):
        h = np.tanh(x @ p["W1"] + p["b1"])
        return h, self._softmax(h @ p["Wa"] + p["ba"]), self._softmax(h @ p["Wb"] + p["bb"])

    def _sgd(self, p, x, ya, yb):
        h, pa, pb = self._forward(p, x)
        ga = pa.copy()
        ga[ya] -= 1
        gb = pb.copy()
        gb[yb] -= 1
        gh = (p["Wa"] @ ga + p["Wb"] @ gb) * (1 - h * h)
        lr, l2 = self.LR, self.L2
        p["Wa"] -= lr * (np.outer(h, ga) + l2 * p["Wa"])
        p["ba"] -= lr * ga
        p["Wb"] -= lr * (np.outer(h, gb) + l2 * p["Wb"])
        p["bb"] -= lr * gb
        p["W1"] -= lr * (np.outer(x, gh) + l2 * p["W1"])
        p["b1"] -= lr * gh

    def predict(self, ctx: Context):
        sd = ctx.sd
        X = self._data(sd)
        state = sd.cache.get("nn")
        if state is None or state[0] > ctx.i:
            state = [0, self._init(X.shape[1])]
            sd.cache["nn"] = state
        p = state[1]
        while state[0] < ctx.i:
            j = state[0]
            v = int(sd.y[j])
            self._sgd(p, X[j], v // 10, v % 10)
            if j > 0:
                for r in np.random.default_rng(1000 + j).integers(0, j, self.REPLAY):
                    rv = int(sd.y[r])
                    self._sgd(p, X[r], rv // 10, rv % 10)
            state[0] += 1
        if ctx.i < 30:
            return UNIFORM.copy()
        if ctx.i < sd.n:
            x = X[ctx.i]
        else:
            F = {k: np.array([ctx.cur[k]]) for k in sd.feature_keys}
            x = _onehot_features(sd, F)[0]
        _, pa, pb = self._forward(p, x)
        return 0.7 * outer(pa, pb) + 0.3 * UNIFORM
