"""Pattern Engine: find the patterns that are real, then predict with them.

Every draw of the four markets is one stream in declaration order (Disawar 05:00,
Faridabad 18:15, Ghaziabad 21:30, Gali 23:30). For each draw and each possible number
the engine computes pattern features from earlier draws only:

    rec0.5 / rec0.8 / rec0.95   short-term repeat: Σ λ^age [number came], last 60 results
    palti                       Σ 0.9^age [palti of the number came]
    pm1 / cut                   ±1 and cut (+5 on both digits) of recent results
    andar_dig / bahar_dig       its tens / units digit came recently in that place
    any_dig                     its digits came recently in any place
    mkt_repeat / mkt_palti      equal to / palti of this market's last result
    weekday                     came in this market on the same weekday (8 weeks)
    dom                         came on the same date of the month (3 months)
    wk_andar / wk_bahar         digit frequency on this weekday in this market (2 years)
    mkt_freq                    count in this market's last 365 results
    balance                     count in the last 1460 results of all markets
    gap                         log(results since it last came)
    sameday                     came earlier the same day

Pattern gate (jodi): at every refit, each pattern gets a score test on the TRAIL draws before
it: t = mean(z of the real number) / standard error, where z is the pattern value
standardised over the 100 numbers. With no pattern t is about 0; a pattern enters the
model only when |t| > GATE_T. Time patterns (weekday, date) are tested the same way.

Model: conditional logit P(v) ∝ exp(Σ w_j z_j(v)) over the gated patterns, fitted by
maximum likelihood on the last TRAIL draws (L2-regularised). Every refit moves the
weights towards what the real numbers had, i.e. it learns from each miss. Andar and
Bahar have their own digit models built the same way; with only ten digit patterns all
of them enter and the fitted weights decide (that tested better than gating them).
"""

from __future__ import annotations

import bisect
import datetime as dt

import numpy as np

from .. import config

WINDOW = 60          # short-term memory: 15 days of four markets
TRAIL = 2920         # draws a refit learns from: about two years of four markets
REFIT = 200          # walk-forward refit interval (live refits after every result)
GATE_T = 2.0         # |t| a jodi pattern needs to enter the model
DIGIT_GATE_T = 0.0   # the 10 digit patterns all enter; their weights decide (tested best)
START = 400          # draws of history before the first prediction
L2 = 1.0             # ridge penalty on the weights (summed log-likelihood)
NEWTON = 8           # Newton steps per refit

_V = np.arange(100)
_REV = (_V % 10) * 10 + _V // 10
_CUT = ((_V // 10 + 5) % 10) * 10 + (_V % 10 + 5) % 10

JODI = ["rec0.5", "rec0.8", "rec0.95", "palti", "pm1", "cut", "andar_dig", "bahar_dig", "any_dig",
        "mkt_repeat", "mkt_palti", "weekday", "dom", "wk_andar", "wk_bahar", "mkt_freq", "balance",
        "gap", "sameday"]
DIGIT = ["d_rec0.5", "d_rec0.8", "d_rec0.95", "d_other_place", "d_balance", "d_gap",
         "d_mkt_last", "d_mkt_last_other", "d_weekday", "d_sameday"]

# Engine 5: "previous result -> next result" transitions (log of smoothed past frequency).
# Jodi only: the same idea for the Andar/Bahar digit models tested lower (51.5 / 52.8 %
# vs 51.8 / 53.1 %), so the digit models stay as they are.
TRANS_JODI = ["tr_market", "tr_stream", "tr_digit_same", "tr_digit_cross"]

LABELS = {
    "tr_market": "Transition: isi market me pichle number ke baad yeh pehle aaya",
    "tr_stream": "Transition: pichle result (koi bhi market) ke baad yeh pehle aaya",
    "tr_digit_same": "Transition: digit → usi jagah ka digit",
    "tr_digit_cross": "Transition: digit → doosri jagah ka digit",
    "rec0.5": "Haal ka number dobara (pichle 2-3 result)",
    "rec0.8": "Haal ka number dobara (pichle ~5 result)",
    "rec0.95": "Haal ka number dobara (pichle ~20 result)",
    "palti": "Haal ke number ki palti",
    "pm1": "Haal ke number ka ±1",
    "cut": "Haal ke number ka cut (+5)",
    "andar_dig": "Andar digit haal me aaya",
    "bahar_dig": "Bahar digit haal me aaya",
    "any_dig": "Dono digit haal me kahin aaye",
    "mkt_repeat": "Isi market ka pichla number",
    "mkt_palti": "Isi market ke pichle number ki palti",
    "weekday": "Time: isi weekday pichle 8 hafte",
    "dom": "Time: isi tareekh pichle 3 mahine",
    "wk_andar": "Time: is weekday ka andar digit (2 saal)",
    "wk_bahar": "Time: is weekday ka bahar digit (2 saal)",
    "mkt_freq": "Is market me 1 saal ki ginti",
    "balance": "Balancing: 4 markets me 1460 result ki ginti",
    "gap": "Gap: kitne result se nahi aaya",
    "sameday": "Aaj pehle aaye market me aaya",
    "d_rec0.5": "Digit haal me aaya (2-3 result)",
    "d_rec0.8": "Digit haal me aaya (~5 result)",
    "d_rec0.95": "Digit haal me aaya (~20 result)",
    "d_other_place": "Digit doosri jagah haal me aaya",
    "d_balance": "Balancing: digit ki 1460 result ki ginti",
    "d_gap": "Gap: digit kitne result se nahi aaya",
    "d_mkt_last": "Isi market ke pichle result me isi jagah",
    "d_mkt_last_other": "Isi market ke pichle result me doosri jagah",
    "d_weekday": "Time: isi weekday digit ki ginti (2 saal)",
    "d_sameday": "Aaj pehle aaye market me digit",
}


class Stream:
    """All results of the four markets as one list in declaration order."""

    def __init__(self, table: dict):
        self.slot = {m: config.MARKETS[m]["result_time"] for m in config.MARKET_KEYS}
        rows = sorted((d, self.slot[m], m, int(v)) for m in config.MARKET_KEYS
                      for d, v in table.get(m, {}).items())
        self.keys = [(d, s) for d, s, _, _ in rows]
        self.dates = [d for d, *_ in rows]
        self.markets = [m for _, _, m, _ in rows]
        self.vals = np.array([v for *_, v in rows], dtype=int)
        self.pos = {(m, d): i for i, (d, _, m, _) in enumerate(rows)}
        self.ord = np.array([d.toordinal() for d in self.dates], dtype=int)
        self.day = np.array([d.day for d in self.dates], dtype=int)
        self.by_market = {m: np.array([i for i, x in enumerate(self.markets) if x == m], dtype=int)
                          for m in config.MARKET_KEYS}

    def __len__(self):
        return len(self.vals)

    def index_of(self, market: str, date: dt.date) -> int:
        i = self.pos.get((market, date))
        return i if i is not None else bisect.bisect_left(self.keys, (date, self.slot[market]))

    def market_history(self, market: str, i: int) -> np.ndarray:
        """Stream positions of this market's results before position i."""
        idx = self.by_market[market]
        return idx[:int(np.searchsorted(idx, i))]

    def since(self, i: int, first_ord: int) -> int:
        """First stream position before i whose date is on/after the ordinal first_ord."""
        return int(np.searchsorted(self.ord[:i], first_ord))


def _gap(recent_newest_first: np.ndarray, size: int, cap: int) -> np.ndarray:
    """Draws since each value last came (1 = the latest draw), capped."""
    seen = np.full(size, cap)
    if len(recent_newest_first):
        u, first = np.unique(recent_newest_first, return_index=True)
        seen[u] = first + 1
    return seen


def _zs(X: np.ndarray) -> np.ndarray:
    """Standardise every pattern over the numbers (axis -2)."""
    m = X.mean(axis=-2, keepdims=True)
    s = X.std(axis=-2, keepdims=True)
    return (X - m) / np.where(s > 0, s, 1)


def jodi_features(st: Stream, i: int, market: str, date: dt.date) -> np.ndarray:
    """(100, len(JODI)) raw pattern values for a draw of `market` on `date` at position i."""
    vals = st.vals
    past = vals[max(0, i - WINDOW):i][::-1]
    age = np.arange(len(past))
    f = np.zeros((100, len(JODI)))
    col = {n: k for k, n in enumerate(JODI)}
    for lam in (0.5, 0.8, 0.95):
        f[:, col[f"rec{lam}"]] = np.bincount(past, weights=lam ** age, minlength=100)
    w = 0.9 ** age
    f[:, col["palti"]] = np.bincount(_REV[past], weights=w, minlength=100)
    f[:, col["pm1"]] = (np.bincount((past + 1) % 100, weights=w, minlength=100)
                        + np.bincount((past - 1) % 100, weights=w, minlength=100))
    f[:, col["cut"]] = np.bincount(_CUT[past], weights=w, minlength=100)
    dt_ = np.bincount(past // 10, weights=w, minlength=10)
    du = np.bincount(past % 10, weights=w, minlength=10)
    f[:, col["andar_dig"]] = dt_[_V // 10]
    f[:, col["bahar_dig"]] = du[_V % 10]
    dany = dt_ + du
    f[:, col["any_dig"]] = dany[_V // 10] + dany[_V % 10]
    o = date.toordinal()
    hist = st.market_history(market, i)
    if len(hist):
        last = vals[hist[-1]]
        f[last, col["mkt_repeat"]] = 1
        f[_REV[last], col["mkt_palti"]] = 1
        h = hist[-730:]
        days = o - st.ord[h]
        same_wd = h[days % 7 == 0]
        recent_wd = h[(days % 7 == 0) & (days <= 56)]
        f[:, col["weekday"]] = np.bincount(vals[recent_wd], minlength=100)
        f[:, col["wk_andar"]] = np.bincount(vals[same_wd] // 10, minlength=10)[_V // 10]
        f[:, col["wk_bahar"]] = np.bincount(vals[same_wd] % 10, minlength=10)[_V % 10]
        f[:, col["mkt_freq"]] = np.bincount(vals[hist[-365:]], minlength=100)
    lo = st.since(i, o - 95)
    dm = np.arange(lo, i)
    dm = dm[(st.day[dm] == date.day) & (st.ord[dm] != o)]
    f[:, col["dom"]] = np.bincount(vals[dm], minlength=100)
    f[:, col["balance"]] = np.bincount(vals[max(0, i - 1460):i], minlength=100)
    f[:, col["gap"]] = np.log1p(_gap(vals[max(0, i - 400):i][::-1], 100, 400))
    today = np.arange(st.since(i, o), i)
    f[:, col["sameday"]] = np.bincount(vals[today], minlength=100)
    return f


def digit_features(st: Stream, i: int, market: str, date: dt.date, place: str) -> np.ndarray:
    """(10, len(DIGIT)) raw pattern values of the andar ('a') or bahar ('b') digit."""
    vals = st.vals
    own_all, oth_all = (vals // 10, vals % 10) if place == "a" else (vals % 10, vals // 10)
    lo = max(0, i - WINDOW)
    own, oth = own_all[lo:i][::-1], oth_all[lo:i][::-1]
    age = np.arange(len(own))
    f = np.zeros((10, len(DIGIT)))
    for c, lam in enumerate((0.5, 0.8, 0.95)):
        f[:, c] = np.bincount(own, weights=lam ** age, minlength=10)
    f[:, 3] = np.bincount(oth, weights=0.9 ** age, minlength=10)
    f[:, 4] = np.bincount(own_all[max(0, i - 1460):i], minlength=10)
    f[:, 5] = np.log1p(_gap(own_all[max(0, i - 200):i][::-1], 10, 200))
    o = date.toordinal()
    hist = st.market_history(market, i)
    if len(hist):
        f[own_all[hist[-1]], 6] = 1
        f[oth_all[hist[-1]], 7] = 1
        h = hist[-730:]
        f[:, 8] = np.bincount(own_all[h[(o - st.ord[h]) % 7 == 0]], minlength=10)
    today = np.arange(st.since(i, o), i)
    f[:, 9] = np.bincount(own_all[today], minlength=10)
    return f


def gate(zy: np.ndarray) -> np.ndarray:
    """Score-test t of every pattern from the real numbers' standardised values (draws × patterns)."""
    if len(zy) < 30:
        return np.zeros(zy.shape[1])
    se = zy.std(axis=0) / np.sqrt(len(zy))
    return np.where(se > 0, zy.mean(axis=0) / np.where(se > 0, se, 1), 0.0)


def _loglik(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    s = X @ w
    s -= s.max(axis=1, keepdims=True)
    return float((s[np.arange(len(y)), y] - np.log(np.exp(s).sum(axis=1))).sum() - 0.5 * L2 * w @ w)


def fit(Z: np.ndarray, y: np.ndarray, cols: list[int], w0: np.ndarray | None = None) -> np.ndarray:
    """Maximum-likelihood weights of the conditional logit on the gated patterns.

    Newton's method with a line search (the patterns are strongly correlated, so plain
    gradient steps can settle on wrong signs); L2 keeps the weights finite.
    """
    X = Z[:, :, cols]
    k = len(cols)
    w = np.zeros(k) if w0 is None else w0.copy()
    n = len(y)
    if n == 0 or not k:
        return w
    rows = np.arange(n)
    f = _loglik(X, y, w)
    for _ in range(NEWTON):
        s = X @ w
        s -= s.max(axis=1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(axis=1, keepdims=True)
        xbar = np.einsum("nv,nvk->nk", p, X)
        g = (X[rows, y] - xbar).sum(axis=0) - L2 * w
        Xs = (X * np.sqrt(p)[:, :, None]).reshape(-1, k)
        H = Xs.T @ Xs - xbar.T @ xbar + L2 * np.eye(k)
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-4:
            f_new = _loglik(X, y, w + t * step)
            if f_new >= f:
                break
            t /= 2
        if t <= 1e-4:
            break
        w, f = w + t * step, f_new
        if np.abs(t * step).max() < 1e-6:
            break
    return w


def probs(z: np.ndarray, cols: list[int], w: np.ndarray) -> np.ndarray:
    s = z[:, cols] @ w if cols else np.zeros(z.shape[0])
    p = np.exp(s - s.max())
    return p / p.sum()


class Part:
    """One model (jodi, andar or bahar): features of every draw, walk-forward predictions."""

    def __init__(self, name: str, Z: np.ndarray, y: np.ndarray, names: list[str], gate_t: float = GATE_T):
        self.name, self.Z, self.y, self.names, self.gate_t = name, Z, y, names, gate_t
        self.ZY = Z[np.arange(len(y)), y] if len(y) else np.zeros((0, len(names)))
        n = len(y)
        self.P = np.full((n, Z.shape[1]), 1.0 / Z.shape[1])    # walk-forward probabilities
        self.cols_at = [None] * n
        self.w_at = [None] * n
        cols, w = [], np.zeros(0)
        for i in range(START, n):
            if (i - START) % REFIT == 0:
                cols, w = self.refit(i)
            self.P[i] = probs(Z[i], cols, w)
            self.cols_at[i], self.w_at[i] = cols, w

    def refit(self, i: int) -> tuple[list[int], np.ndarray]:
        lo = max(0, i - TRAIL)
        t = gate(self.ZY[lo:i])
        cols = [j for j in range(len(self.names)) if abs(t[j]) > self.gate_t]
        return cols, fit(self.Z[lo:i], self.y[lo:i], cols)

    def scan(self, i: int) -> list[dict]:
        """Every pattern's test on the TRAIL draws before i (what the next refit sees)."""
        lo = max(0, i - TRAIL)
        t = gate(self.ZY[lo:i])
        cols, w = self.refit(i)
        wt = dict(zip(cols, w))
        return [{"name": n, "label": LABELS.get(n, n), "t": round(float(t[j]), 2), "used": j in wt,
                 "weight": round(float(wt.get(j, 0.0)), 4)} for j, n in enumerate(self.names)]


class Transitions:
    """How often each result followed each previous result, from earlier draws only.

    Tables: the stream (previous draw of any market -> next), each market (its own
    previous result -> its next) and the stream's digits (a->a, b->b, a->b, b->a). Every
    cell starts at 0.5 (smoothing), so a jump never seen before is not impossible, just unseen.
    """

    def __init__(self):
        self.stream = np.full((100, 100), 0.5)
        self.market = {m: np.full((100, 100), 0.5) for m in config.MARKET_KEYS}
        self.dig = np.full((4, 10, 10), 0.5)

    def add(self, prev_stream: int | None, prev_market: int | None, market: str, y: int) -> None:
        if prev_stream is not None:
            self.stream[prev_stream, y] += 1
            pa, pb, a, b = prev_stream // 10, prev_stream % 10, y // 10, y % 10
            self.dig[0, pa, a] += 1
            self.dig[1, pb, b] += 1
            self.dig[2, pa, b] += 1
            self.dig[3, pb, a] += 1
        if prev_market is not None:
            self.market[market][prev_market, y] += 1

    @staticmethod
    def previous(st: "Stream", i: int, market: str) -> tuple[int | None, int | None]:
        hist = st.market_history(market, i)
        return (int(st.vals[i - 1]) if i > 0 else None), (int(st.vals[hist[-1]]) if len(hist) else None)

    @classmethod
    def upto(cls, st: "Stream", i: int) -> "Transitions":
        tr = cls()
        for j in range(i):
            ps, pm = cls.previous(st, j, st.markets[j])
            tr.add(ps, pm, st.markets[j], int(st.vals[j]))
        return tr

    @staticmethod
    def _log_row(t: np.ndarray, r: int) -> np.ndarray:
        return np.log(t[r] / t[r].sum())

    def jodi(self, prev_stream: int | None, prev_market: int | None, market: str) -> np.ndarray:
        f = np.zeros((100, len(TRANS_JODI)))
        if prev_market is not None:
            f[:, 0] = self._log_row(self.market[market], prev_market)
        if prev_stream is not None:
            f[:, 1] = self._log_row(self.stream, prev_stream)
            pa, pb = prev_stream // 10, prev_stream % 10
            la, lb = self._log_row(self.dig[0], pa), self._log_row(self.dig[1], pb)
            f[:, 2] = la[_V // 10] + lb[_V % 10]
            ca, cb = self._log_row(self.dig[2], pa), self._log_row(self.dig[3], pb)
            f[:, 3] = ca[_V % 10] + cb[_V // 10]
        return f


class PatternEngine:
    """Jodi, Andar and Bahar pattern models over the whole stream.

    transitions=True adds the jodi transition patterns (Engine 5)."""

    def __init__(self, table: dict, transitions: bool = False):
        self.st = st = Stream(table)
        self.transitions = transitions
        self.jodi_names = JODI + (TRANS_JODI if transitions else [])
        self.digit_names = DIGIT
        n = len(st)
        Fj = np.zeros((n, 100, len(self.jodi_names)))
        Fa = np.zeros((n, 10, len(self.digit_names)))
        Fb = np.zeros((n, 10, len(self.digit_names)))
        tr = Transitions() if transitions else None
        for i in range(n):
            m, d = st.markets[i], st.dates[i]
            Fj[i] = self._jodi_raw(i, m, d, tr)
            Fa[i] = digit_features(st, i, m, d, "a")
            Fb[i] = digit_features(st, i, m, d, "b")
            if tr is not None:
                ps, pm = Transitions.previous(st, i, m)
                tr.add(ps, pm, m, int(st.vals[i]))
        self._tr_all = tr                                   # tables with every result
        y = st.vals
        self.jodi = Part("jodi", _zs(Fj), y, self.jodi_names)
        self.andar = Part("andar", _zs(Fa), y // 10, self.digit_names, DIGIT_GATE_T)
        self.bahar = Part("bahar", _zs(Fb), y % 10, self.digit_names, DIGIT_GATE_T)

    def _jodi_raw(self, i, market, date, tr):
        f = jodi_features(self.st, i, market, date)
        if tr is None:
            return f
        return np.concatenate([f, tr.jodi(*Transitions.previous(self.st, i, market), market)], axis=1)

    def next(self, market: str, date: dt.date) -> dict:
        """Prediction for a draw that has no result yet, learned from every result so far."""
        st = self.st
        i = st.index_of(market, date)
        tr = None
        if self.transitions:
            tr = self._tr_all if i >= len(st) else Transitions.upto(st, i)
        out = {"position": i}
        for part, raw in ((self.jodi, self._jodi_raw(i, market, date, tr)),
                          (self.andar, digit_features(st, i, market, date, "a")),
                          (self.bahar, digit_features(st, i, market, date, "b"))):
            cols, w = part.refit(min(i, len(part.y)))
            out[part.name] = probs(_zs(raw), cols, w)
            out[part.name + "_used"] = {part.names[c]: round(float(x), 4) for c, x in zip(cols, w)}
        return out

    def contributions(self, i: int, v: int) -> list[tuple[str, float, float]]:
        """(pattern, push, value) for number v at draw i: push > 0 moved it up the list;
        value > 0 means v had more of that pattern than the average number."""
        cols, w = self.jodi.cols_at[i] or [], self.jodi.w_at[i]
        if w is None or not cols:
            return []
        z = self.jodi.Z[i]
        rel = z[v, cols] - (self.jodi.P[i] @ z[:, cols])
        return sorted(((self.jodi.names[c], float(wk * r), float(r)) for c, wk, r in zip(cols, w, rel)),
                      key=lambda x: -abs(x[1]))
