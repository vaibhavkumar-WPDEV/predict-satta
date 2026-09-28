"""Engine 4.0 Top-10: cross-market recency that re-tunes itself after every miss.

The one pattern that survived five years of walk-forward tests: a number that came up
recently in ANY of the four markets comes again a little more often than chance (the
last 3 draws repeat 1.22x as often as random). The 30-model ensemble diluted it
(10.5 % Top-10 over 2021-26); ranking by it directly gives about 11.5 %.

    score(v) = sum over the last WINDOW draws of all markets (same-day markets that were
               declared earlier included) of  lam^age * ([v came] + palti_w * [palti of v came])

Twelve settings (lam, palti_w) run side by side. After every result each setting is
marked HIT or MISS (was the real number in its Top-10?), and the next list uses the
setting with the most hits over the last TRAIL draws. So every miss moves the choice
towards the settings that would have caught it; nothing is tuned on future data.
"""

from __future__ import annotations

import bisect
import datetime as dt

import numpy as np

from .. import config

LAMS = (0.5, 0.7, 0.8, 0.85, 0.9, 0.95)
PALTI = (0.0, 0.3)
CANDS = [(lam, pw) for lam in LAMS for pw in PALTI]
WINDOW = 60                  # draws looked back: 15 days of four markets
TRAIL = 1460                 # draws a setting is judged on: about a year of four markets
MU_GRID = np.array([0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.3, 0.4])
_TIE = np.arange(100) * 1e-12  # equal scores: lower number first, like rank_of
_REV = np.array([(v % 10) * 10 + v // 10 for v in range(100)])


def label(c: int) -> str:
    lam, pw = CANDS[c]
    return f"Recency (λ={lam}{', palti ' + str(pw) if pw else ''})"


class Recency:
    """All four markets as one chronological stream, scored walk-forward."""

    def __init__(self, table: dict):
        slot = {m: config.MARKETS[m]["result_time"] for m in config.MARKET_KEYS}
        self.stream = sorted((d, slot[m], m, int(v)) for m in config.MARKET_KEYS
                             for d, v in table.get(m, {}).items())
        self.keys = [(d, s) for d, s, _, _ in self.stream]
        self.vals = np.array([v for *_, v in self.stream], dtype=int)
        self.pos = {(m, d): i for i, (d, _, m, _) in enumerate(self.stream)}
        self.slot = slot
        n = len(self.vals)
        self.H = np.zeros((n, len(CANDS)), bool)    # was draw i in setting c's Top-10
        self.Q = np.zeros((n, len(CANDS)))          # share of setting c's score draw i had
        for i in range(n):
            y = self.vals[i]
            for c in range(len(CANDS)):
                s = self._scores(i, c)
                tot = s.sum()
                if tot > 0:
                    self.Q[i, c] = s[y] / tot
                    self.H[i, c] = y in np.argsort(-(s - _TIE), kind="stable")[:10]

    def _scores(self, i: int, c: int) -> np.ndarray:
        lam, pw = CANDS[c]
        past = self.vals[max(0, i - WINDOW):i][::-1]
        if not len(past):
            return np.zeros(100)
        w = lam ** np.arange(len(past))
        s = np.bincount(past, weights=w, minlength=100)
        if pw:
            s = s + pw * np.bincount(_REV[past], weights=w, minlength=100)
        return s

    def choice(self, i: int) -> int:
        """Setting with the most Top-10 hits over the TRAIL draws before draw i."""
        return int(np.argmax(self.H[max(0, i - TRAIL):i].sum(axis=0)))

    def ranking(self, i: int) -> list[tuple[int, float]]:
        h = self.H[max(0, i - TRAIL):i]
        rates = h.mean(axis=0) if len(h) else np.zeros(len(CANDS))
        return sorted(((c, float(r)) for c, r in enumerate(rates)), key=lambda x: -x[1])

    def dist(self, i: int) -> tuple[np.ndarray, int]:
        """Distribution for the draw at stream position i, and the setting used."""
        c = self.choice(i)
        s = self._scores(i, c)
        tot = s.sum()
        if tot == 0:
            return np.full(100, 0.01), c
        q = self.Q[max(0, i - TRAIL):i, c]
        # how much of the probability to give the score: maximum likelihood on past draws
        mu = MU_GRID[int(np.argmax([np.log((1 - m) / 100 + m * q).sum() for m in MU_GRID]))] if len(q) else 0.1
        p = (1 - mu) / 100 + mu * s / tot
        return p / p.sum(), c

    def index_of(self, market: str, date: dt.date) -> int:
        """Stream position of (market, date): its own entry, or where it would be inserted."""
        i = self.pos.get((market, date))
        return i if i is not None else bisect.bisect_left(self.keys, (date, self.slot[market]))
