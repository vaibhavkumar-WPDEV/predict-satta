"""Symbolic layer: numerology and the moon's tithi as *hypotheses*.

These are not established science. They are turned into concrete rules
("result = the date's mulank twice", "result = today's tithi", ...), every
rule is scored on the real history, and the ensemble only trusts a rule as
much as it actually worked. Weather and tarot are not used: there is no
weather data in the results, and a tarot draw is itself random, so it can not
carry information about another random draw.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from .base import rev

SYNODIC_MONTH = 29.530588853
# a new moon: 2000-01-06 18:14 UTC (days counted from the calendar date, noon IST)
_NEW_MOON = dt.datetime(2000, 1, 6, 18, 14)


def moon_age(date: dt.date) -> float:
    """Days since the last new moon at noon IST on `date` (0 .. 29.53)."""
    t = dt.datetime.combine(date, dt.time(6, 30))  # 12:00 IST in UTC
    return ((t - _NEW_MOON).total_seconds() / 86400.0) % SYNODIC_MONTH


def tithi(date: dt.date) -> int:
    """Lunar day 1..30 (1-15 shukla paksha, 16-30 krishna paksha)."""
    return int(moon_age(date) / SYNODIC_MONTH * 30) + 1


def root(n: int) -> int:
    """Mulank / digital root 1..9 (0 for 0)."""
    return 0 if n == 0 else 1 + (n - 1) % 9


def candidates(date: dt.date) -> dict[str, int]:
    """Numbers a numerology/tithi reading would point to for `date`."""
    d, m, y = date.day, date.month, date.year % 100
    full = sum(int(c) for c in date.strftime("%d%m%Y"))
    t = tithi(date)
    return {
        "tareekh (DD)": d,
        "ulta tareekh": int(rev(d)),
        "mahina (MM)": m,
        "saal (YY)": y,
        "DD+MM": (d + m) % 100,
        "DD−MM": (d - m) % 100,
        "DD×MM": (d * m) % 100,
        "DD+YY": (d + y) % 100,
        "mulank x2": root(d) * 11,
        "bhagyank x2": root(full) * 11,
        "mulank-bhagyank": root(d) * 10 + root(full) % 10,
        "tithi": t % 100,
        "ulta tithi": int(rev(t % 100)),
        "tithi mulank x2": root(t) * 11,
    }


RULES = list(candidates(dt.date(2026, 1, 1)))


def rule_table(dates: list[dt.date]) -> np.ndarray:
    """(len(dates), len(RULES)) matrix of each rule's number for each date."""
    return np.array([[candidates(d)[r] for r in RULES] for d in dates], dtype=int).reshape(len(dates), len(RULES))
