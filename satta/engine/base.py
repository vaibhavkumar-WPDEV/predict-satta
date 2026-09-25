"""Shared helpers: number transforms, aligned series, prediction context.

A jodi is a number 00-99. Andar = tens digit, Bahar = units digit.
Every expert returns a probability vector of length 100 (index = jodi).
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from .. import config

N = 100
UNIFORM = np.full(N, 1.0 / N)
UNIFORM10 = np.full(10, 0.1)


def normalize(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 0.0, None)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full(len(p), 1.0 / len(p))
    return p / s


def outer(pt, pu) -> np.ndarray:
    """P(jodi) from independent Andar/Bahar digit distributions."""
    return normalize(np.outer(normalize(pt), normalize(pu)).ravel())


def andar_bahar(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(p).reshape(10, 10)
    return m.sum(axis=1), m.sum(axis=0)


def rev(x):
    """Palti / ulta: 47 -> 74."""
    return (x % 10) * 10 + x // 10


def cut(x):
    """Cut: add 5 to each digit (mod 10): 47 -> 92."""
    return ((x // 10 + 5) % 10) * 10 + (x % 10 + 5) % 10


def latest_before(series: dict, date: dt.date, max_gap: int = 3) -> int:
    for k in range(1, max_gap + 1):
        v = series.get(date - dt.timedelta(days=k))
        if v is not None:
            return int(v)
    return -1


class SeriesData:
    """One market's result series plus the features known before each draw.

    F[key][i] is the value of feature `key` known *before* draw i:
      A1/A2  previous two draws of the same market
      <mkt>  the other market's result of the previous calendar day(s)
      DM/WD/MO  day of month, weekday, month of draw i
    Missing values are -1.
    """

    def __init__(self, table: dict[str, dict[dt.date, int]], market: str):
        self.market = market
        self.table = table
        self.cache: dict = {}
        series = table.get(market, {})
        self.dates: list[dt.date] = sorted(series)
        self.y = np.array([series[d] for d in self.dates], dtype=int)
        self.n = len(self.y)
        self.others = [m for m in config.MARKET_KEYS if m != market]
        self.feature_keys = ["A1", "A2", "DM", "WD", "MO"] + self.others
        feats = [self.features_for(d, i) for i, d in enumerate(self.dates)]
        self.F = {k: np.array([f[k] for f in feats], dtype=int) if feats else np.zeros(0, int)
                  for k in self.feature_keys}

    def features_for(self, date: dt.date, i: int) -> dict[str, int]:
        f = {
            "A1": int(self.y[i - 1]) if i >= 1 else -1,
            "A2": int(self.y[i - 2]) if i >= 2 else -1,
            "DM": date.day,
            "WD": date.weekday(),
            "MO": date.month,
        }
        for m in self.others:
            f[m] = latest_before(self.table.get(m, {}), date)
        return f

    def context(self, i: int, date: dt.date | None = None) -> "Context":
        """Context for predicting draw i (i == n means the next, unknown draw)."""
        if i < self.n:
            date = self.dates[i]
            cur = {k: int(self.F[k][i]) for k in self.feature_keys}
        else:
            cur = self.features_for(date, i)
        return Context(self, i, date, cur)


class Context:
    """Everything an expert may look at: strictly data from before draw i."""

    def __init__(self, sd: SeriesData, i: int, date: dt.date, cur: dict[str, int]):
        self.sd = sd
        self.i = i
        self.date = date
        self.cur = cur
        self.y = sd.y[:i]

    def past(self, key: str) -> np.ndarray:
        return self.sd.F[key][: self.i]
