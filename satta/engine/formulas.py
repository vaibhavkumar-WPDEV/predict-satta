"""Formula discovery: the engine writes its own modular-arithmetic formulas.

Search space (every value is known before the draw):

    jodi   = (c1*U + c2*V + k) mod 100      U, V in {pichla, 2 din pehle,
                                             other markets' kal, their ulta, tareekh}
    andar  = (c1*u + c2*v + k) mod 10       u, v digits of the above + tareekh,
    bahar  = (c1*u + c2*v + k) mod 10       hafte ka din, mahina

with c1, c2 in {+1, -1} and k fitted as the most frequent residual
(k = argmax_k #{t : (y_t - c1 U_t - c2 V_t) mod m == k}).

Hundreds of formulas are tried, so the best one *always* looks good on the
data it was found on. Every formula is therefore judged on data it has never
seen (walk-forward holdout) against pure chance with an exact binomial test.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .. import config
from .base import SeriesData, rev
from .stats import binom_sf


def _short(m: str) -> str:
    return config.MARKETS[m]["short"]


def _safe(fn, x):
    x = np.asarray(x)
    return np.where(x >= 0, fn(np.where(x >= 0, x, 0)), -1)


def variables(sd: SeriesData, F: dict) -> tuple[dict, dict, dict]:
    """(jodi_vars, digit_vars, names) from feature arrays (or 1-element arrays)."""
    me = _short(sd.market)
    jv, names = {}, {}
    jv["A1"], names["A1"] = F["A1"], f"{me}(pichla)"
    jv["A1r"], names["A1r"] = _safe(rev, F["A1"]), f"ulta {me}(pichla)"
    jv["A2"], names["A2"] = F["A2"], f"{me}(2 din pehle)"
    for m in sd.others:
        jv[m], names[m] = F[m], f"{_short(m)}(kal)"
        jv[m + "r"], names[m + "r"] = _safe(rev, F[m]), f"ulta {_short(m)}(kal)"
    jv["DM"], names["DM"] = F["DM"], "tareekh"

    dv = {}
    for key in ["A1", "A2"] + sd.others:
        dv[key + ".t"] = _safe(lambda x: x // 10, F[key])
        dv[key + ".u"] = _safe(lambda x: x % 10, F[key])
        names[key + ".t"] = f"andar[{names[key]}]"
        names[key + ".u"] = f"bahar[{names[key]}]"
    dv["DM.t"], names["DM.t"] = np.asarray(F["DM"]) // 10, "tareekh ka 1st digit"
    dv["DM.u"], names["DM.u"] = np.asarray(F["DM"]) % 10, "tareekh ka 2nd digit"
    dv["WD"], names["WD"] = np.asarray(F["WD"]), "hafte ka din(Som=0)"
    dv["MO"], names["MO"] = np.asarray(F["MO"]) % 10, "mahina"
    return jv, dv, names


def _series_vars(sd: SeriesData):
    key = "formula_vars"
    if key not in sd.cache:
        sd.cache[key] = variables(sd, sd.F)
    return sd.cache[key]


def _targets(sd: SeriesData) -> dict[str, tuple[np.ndarray, int]]:
    return {"jodi": (sd.y, 100), "andar": (sd.y // 10, 10), "bahar": (sd.y % 10, 10)}


def search(Y: np.ndarray, V: dict[str, np.ndarray], mod: int, lo: int, hi: int,
           min_n: int = 15) -> list[dict]:
    """All 1- and 2-term formulas on rows [lo, hi), best k fitted, sorted by hits."""
    Y = Y[lo:hi]
    out = []
    vals = {k: v[lo:hi] for k, v in V.items()}

    def fit(terms):
        valid = np.ones(len(Y), bool)
        acc = np.zeros(len(Y), int)
        for c, name in terms:
            x = vals[name]
            valid &= x >= 0
            acc = acc + c * x
        n = int(valid.sum())
        if n < min_n:
            return
        r = (Y[valid] - acc[valid]) % mod
        cnt = np.bincount(r, minlength=mod)
        k = int(cnt.argmax())
        out.append({"terms": terms, "k": k, "mod": mod, "hits": int(cnt[k]), "n": n})

    fit([])
    names = list(vals)
    for u in names:
        for c in (1, -1):
            fit([(c, u)])
    for u, v in combinations(names, 2):
        for c1 in (1, -1):
            for c2 in (1, -1):
                fit([(c1, u), (c2, v)])
    out.sort(key=lambda f: (-f["hits"] / f["n"], len(f["terms"])))
    return out


def apply(f: dict, V: dict[str, np.ndarray]) -> np.ndarray:
    first = next(iter(V.values()))
    acc = np.zeros(len(first), int)
    valid = np.ones(len(first), bool)
    for c, name in f["terms"]:
        x = np.asarray(V[name])
        valid &= x >= 0
        acc = acc + c * x
    return np.where(valid, (acc + f["k"]) % f["mod"], -1)


def expression(f: dict, target: str, names: dict) -> str:
    parts = []
    for c, name in f["terms"]:
        sign = "+" if c > 0 else "−"
        parts.append(f"{sign} {names[name]}")
    body = " ".join(parts).lstrip("+ ").strip()
    if body.startswith("−"):
        body = "−" + body[1:].lstrip()
    k = f["k"]
    body = f"{body} + {k:0{2 if f['mod'] == 100 else 1}d}" if body else f"{k:0{2 if f['mod'] == 100 else 1}d}"
    return f"{target} = ({body}) mod {f['mod']}"


def discover(sd: SeriesData, upto: int, kind: str, top: int) -> list[dict]:
    """Best formulas of one kind using draws [0, upto) only."""
    jv, dv, _ = _series_vars(sd)
    Y, mod = _targets(sd)[kind]
    V = jv if kind == "jodi" else dv
    return search(Y, V, mod, 0, upto)[:top]


def current_vars(sd: SeriesData, cur: dict) -> tuple[dict, dict]:
    F = {k: np.array([cur[k]]) for k in sd.feature_keys}
    jv, dv, _ = variables(sd, F)
    return jv, dv


def lcg_solve(y: np.ndarray, top: int = 3) -> list[tuple[int, int, int, int]]:
    """Aryabhata's kuttaka for y_t ≡ a·y_(t-1) + c (mod 100): all a, best c. Returns (a, c, hits, n)."""
    x, t = y[:-1], y[1:]
    A = np.arange(100)[:, None]
    R = (t[None, :] - A * x[None, :]) % 100
    counts = np.bincount((A * 100 + R).ravel(), minlength=100 * 100)
    best = np.argsort(-counts, kind="stable")[:top]
    return [(int(b // 100), int(b % 100), int(counts[b]), len(t)) for b in best]


def _lcg_report(sd: SeriesData, split: int, top: int, with_next: bool) -> dict:
    Y = sd.y
    me = _short(sd.market)
    rows = []
    for a, c, hits, n in lcg_solve(Y[:split], top):
        pred = (a * Y[split - 1:-1] + c) % 100
        t_hits = int((pred == Y[split:]).sum())
        t_n = len(pred)
        p_val = binom_sf(t_hits, t_n, 0.01)
        rows.append({"formula": f"{me} = ({a}·{me}(pichla) + {c:02d}) mod 100",
                     "train_hits": hits, "train_n": n, "train_rate": hits / n,
                     "test_hits": t_hits, "test_n": t_n, "test_rate": t_hits / t_n,
                     "chance_rate": 0.01, "p_value": p_val, "significant": p_val * top < 0.05})
    nxt = []
    if with_next:
        for a, c, hits, n in lcg_solve(Y, 5):
            nxt.append({"formula": f"{me} = ({a}·{me}(pichla) + {c:02d}) mod 100",
                        "value": int((a * Y[-1] + c) % 100), "hits": hits, "n": n})
    return {"tested": 100 * 100, "chance_rate": 0.01, "top": rows, "next": nxt}


def report(sd: SeriesData, next_cur: dict | None, holdout: float = 0.3, top: int = 8) -> dict:
    """Discover on the first 70 %, verify on the last 30 %, then refit on all data."""
    n = sd.n
    if n < 40:
        return {"ready": False, "reason": f"kam se kam 40 result chahiye (abhi {n})"}
    split = int(n * (1 - holdout))
    jv, dv, names = _series_vars(sd)
    me = _short(sd.market)
    out = {"ready": True, "train_days": split, "test_days": n - split, "kinds": {}}
    for kind, (Y, mod) in _targets(sd).items():
        V = jv if kind == "jodi" else dv
        found = search(Y, V, mod, 0, split)
        tested = len(found)
        p0 = 1.0 / mod
        rows = []
        for f in found[:top]:
            pred = apply(f, {k: v[split:] for k, v in V.items()})
            valid = pred >= 0
            t_n = int(valid.sum())
            t_hits = int((pred[valid] == Y[split:][valid]).sum())
            p_val = binom_sf(t_hits, t_n, p0) if t_n else 1.0
            label = {"jodi": me, "andar": f"andar({me})", "bahar": f"bahar({me})"}[kind]
            rows.append({
                "formula": expression(f, label, names),
                "train_hits": f["hits"], "train_n": f["n"],
                "train_rate": f["hits"] / f["n"],
                "test_hits": t_hits, "test_n": t_n,
                "test_rate": t_hits / t_n if t_n else 0.0,
                "chance_rate": p0,
                "p_value": p_val,
                # top formulas were chosen from `top` candidates on the holdout
                "significant": p_val * top < 0.05,
            })
        live = []
        if next_cur is not None:
            cjv, cdv = current_vars(sd, next_cur)
            CV = cjv if kind == "jodi" else cdv
            for f in search(Y, V, mod, 0, n)[:5]:
                val = int(apply(f, CV)[0])
                if val >= 0:
                    live.append({"formula": expression(f, kind if kind != "jodi" else me, names),
                                 "value": val, "hits": f["hits"], "n": f["n"]})
        out["kinds"][kind] = {"tested": tested, "chance_rate": p0, "top": rows, "next": live}
    out["kinds"]["aryabhata"] = _lcg_report(sd, split, top, next_cur is not None)
    return out
