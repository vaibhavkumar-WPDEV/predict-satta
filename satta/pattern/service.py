"""Pattern Engine loop: read results -> learn -> predict -> lock -> score -> publish.

Reads the shared data/results.csv (engine 4.0's watcher fetches it) and writes only
data/pattern/: predictions.jsonl (hash-locked), dashboard.json, prediction_history.csv.
Lock timing, holidays and arrival times use the same rules as engine 4.0, so both
engines predict each draw with the same information.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np

from .. import config, service as main, storage
from ..engine import recency
from ..engine.base import SeriesData, andar_bahar
from ..engine.ensemble import DIGIT_K, rank_of, summarize
from ..engine.stats import binom_sf
from . import ENGINE
from .engine import DIGIT, GATE_T, JODI, LABELS, REFIT, START, TRAIL, PatternEngine

log = logging.getLogger("satta.pattern")
_SRC = Path(__file__).parent


def pattern_dir() -> Path:
    d = config.DATA_DIR / "pattern"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _code_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(_SRC.glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


CODE_HASH = _code_hash()


# ------------------------------------------------------------ locked predictions

def pattern_hash(pred: dict) -> str:
    """SHA-256 over everything that must not change after locking (digits included)."""
    core = {k: pred[k] for k in ("market", "date", "created_at", "top10", "andar", "bahar", "dist")}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify(pred: dict) -> bool:
    return pred.get("hash") == pattern_hash(pred)


def load_predictions() -> list[dict]:
    path = pattern_dir() / "predictions.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_prediction(pred: dict) -> None:
    with open(pattern_dir() / "predictions.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(pred, ensure_ascii=False) + "\n")


def _top(p: np.ndarray, k: int) -> list[list]:
    order = np.argsort(-p, kind="stable")[:k]
    return [[int(v), round(float(p[v]), 5)] for v in order]


def _digits(p: np.ndarray, k: int = DIGIT_K) -> list[int]:
    return [int(d) for d in np.argsort(-p, kind="stable")[:k]]


def make_prediction(pe: PatternEngine, sd: SeriesData, date: dt.date, now: dt.datetime) -> dict:
    nx = pe.next(sd.market, date)
    rt = config.result_datetime(sd.market, date)
    used = sorted(nx["jodi_used"].items(), key=lambda x: -abs(x[1]))
    pred = {
        "market": sd.market,
        "date": date.isoformat(),
        "created_at": now.astimezone(config.IST).isoformat(timespec="seconds"),
        "result_time": rt.isoformat(timespec="minutes"),
        "late": now > rt,
        "engine": ENGINE,
        "trained_on": int(nx["position"]),
        "last_result": sd.dates[-1].isoformat() if sd.n else None,
        "top10": _top(nx["jodi"], 10),
        "andar": _top(nx["andar"], DIGIT_K),
        "bahar": _top(nx["bahar"], DIGIT_K),
        "dist": [round(float(x), 5) for x in nx["jodi"]],
        "patterns": dict(used),
        "digit_patterns": {"andar": nx["andar_used"], "bahar": nx["bahar_used"]},
        "model": "Pattern Engine: " + ", ".join(LABELS.get(n, n) for n, _ in used[:3]),
        "support": [f"{LABELS.get(n, n)} ({w:+.3f})" for n, w in used],
        "contra": [],
    }
    pred["hash"] = pattern_hash(pred)
    return pred


# ---------------------------------------------------------------------- scoring

def _score(pj: np.ndarray, pa: np.ndarray, pb: np.ndarray, actual: int,
           andar: list[int] | None = None, bahar: list[int] | None = None) -> dict:
    r = rank_of(pj, actual)
    andar = andar if andar is not None else _digits(pa)
    bahar = bahar if bahar is not None else _digits(pb)
    a, b = actual // 10, actual % 10
    return {"rank": r, "hit1": r <= 1, "hit5": r <= 5, "hit10": r <= 10,
            "andar_hit": a in andar, "bahar_hit": b in bahar,
            "andar1_hit": bool(andar) and a == andar[0], "bahar1_hit": bool(bahar) and b == bahar[0],
            "prob": float(pj[actual]), "logloss": -math.log(max(float(pj[actual]), 1e-300))}


def _why(pe: PatternEngine, i: int, sc: dict, andar: list[int], bahar: list[int]) -> list[str]:
    st = pe.st
    y = int(st.vals[i])
    a, b = y // 10, y % 10
    lines = [f"✔ HIT: {y:02d} Top-10 me #{sc['rank']} par tha." if sc["hit10"] else
             f"✘ MISS: {y:02d} ko rank {sc['rank']}/100 mila (probability {sc['prob']:.2%}, random 1.00%).",
             ("✔" if sc["andar_hit"] else "✘") + f" Andar {a} top-{DIGIT_K} ({' '.join(map(str, andar))}) me "
             + ("tha." if sc["andar_hit"] else "nahi tha."),
             ("✔" if sc["bahar_hit"] else "✘") + f" Bahar {b} top-{DIGIT_K} ({' '.join(map(str, bahar))}) me "
             + ("tha." if sc["bahar_hit"] else "nahi tha.")]
    before = st.vals[max(0, i - 400):i][::-1]
    seen = np.flatnonzero(before == y)
    count = int((st.vals[max(0, i - 1460):i] == y).sum())
    avg = min(i, 1460) / 100
    lines.append(f"{y:02d} pichli baar {int(seen[0]) + 1} result pehle aaya tha" if len(seen)
                 else f"{y:02d} pichle {len(before)} result me nahi aaya tha")
    lines[-1] += f"; pichle {min(i, 1460)} result me {count} baar (ausat {avg:.1f})."
    contrib = pe.contributions(i, y)

    def said(n, r):
        return f"{LABELS.get(n, n)} ({'ausat se zyada' if r > 0 else 'ausat se kam'})"

    ups = [said(n, r) for n, c, r in contrib if c > 0.005][:2]
    downs = [said(n, r) for n, c, r in contrib if c < -0.005][:2]
    if ups:
        lines.append("Is number ko upar laaye: " + "; ".join(ups) + ".")
    if downs:
        lines.append("Is number ko neeche kiya: " + "; ".join(downs) + ".")
    return lines


def _lesson(pred: dict, now_used: dict) -> str | None:
    """How the jodi pattern weights moved since this prediction was locked."""
    old = pred.get("patterns") or {}
    names = set(old) | set(now_used)
    if not names:
        return None
    moves = sorted(((n, old.get(n, 0.0), now_used.get(n, 0.0)) for n in names),
                   key=lambda x: -abs(x[2] - x[1]))[:2]
    parts = [f"{LABELS.get(n, n)} {o:+.3f} → {w:+.3f}" for n, o, w in moves]
    return "Galti se seekha (is result ke baad weights dobara fit hue): " + "; ".join(parts) + "."


def _summaries(scores: list[dict]) -> dict:
    return {"all": main._clean(summarize(scores)), "last200": main._clean(summarize(scores[-200:])),
            "last100": main._clean(summarize(scores[-100:])), "last30": main._clean(summarize(scores[-30:])),
            "last7": main._clean(summarize(scores[-7:]))}


def _decision(hits: np.ndarray) -> dict | None:
    n = len(hits)
    if n < 100:
        return None
    k = int(hits.sum())
    p = binom_sf(k, n, 0.1)
    r1, r2 = float(hits[: n // 2].mean()), float(hits[n // 2:].mean())
    level = main.signal_quality(k / n, p, r1, r2)
    recent = hits[-730:]
    rate = float(recent.mean())
    lo, hi = main._wilson(int(recent.sum()), len(recent))
    q05, q95 = main._binom_quantiles(30, rate)
    edge = level in ("Strong", "Moderate")
    r05, r95 = main._binom_quantiles(30, 0.1)
    return {
        "verdict": "EDGE" if edge else "NO EDGE", "level": level,
        "confidence": {"Strong": "High", "Moderate": "Medium", "Weak": "Low", "Unreliable": "None"}[level],
        "text": ("Tested edge: Pattern Engine ki Top-10 random se behtar sabit hui hai." if edge else
                 "Insufficient predictive edge: is market me Top-10 random (10%) se behtar sabit nahi hui."),
        "prob": main._r(rate), "prob_lo": main._r(lo), "prob_hi": main._r(hi), "prob_days": int(len(recent)),
        "all_rate": main._r(k / n), "all_n": n, "p_value": main._r(p, 6), "half1": main._r(r1), "half2": main._r(r2),
        "mc30": {"expected": main._r(30 * rate, 1), "lo": q05, "hi": q95,
                 "random": {"expected": 3.0, "lo": r05, "hi": r95}},
    }


def _coverage(ranks: np.ndarray, a_ranks: np.ndarray, b_ranks: np.ndarray) -> dict:
    last = ranks[-200:]
    grid = (a_ranks <= DIGIT_K) & (b_ranks <= DIGIT_K)   # jodi made of Andar top-5 x Bahar top-5
    return {
        "days": len(ranks),
        "jodi": [{"n": n, "tested": main._r((ranks <= n).mean()), "tested_200": main._r((last <= n).mean()),
                  "random": n / 100} for n in (1, 5, 10, 20, 25, 30, 40, 50)],
        "grid": {"n": DIGIT_K * DIGIT_K, "tested": main._r(grid.mean()), "tested_200": main._r(grid[-200:].mean()),
                 "random": DIGIT_K * DIGIT_K / 100},
        "andar": [{"k": k, "tested": main._r((a_ranks <= k).mean()), "random": k / 10} for k in (1, 2, 3, 5)],
        "bahar": [{"k": k, "tested": main._r((b_ranks <= k).mean()), "random": k / 10} for k in (1, 2, 3, 5)],
    }


def _live(market: str, preds: list[dict], table: dict, pe: PatternEngine, now: dt.datetime,
          now_used: dict) -> dict:
    rows, scores, counted = [], [], []
    for p in sorted((p for p in preds if p["market"] == market), key=lambda p: p["date"], reverse=True):
        d = dt.date.fromisoformat(p["date"])
        actual = table[market].get(d)
        row = {"date": p["date"], "created_at": p["created_at"], "late": p.get("late", False),
               "engine": p.get("engine"), "hash": p["hash"], "verified": verify(p),
               "top10": [v for v, _ in p["top10"]], "andar": [x for x, _ in p["andar"]],
               "bahar": [x for x, _ in p["bahar"]], "locked_digits": len(p["andar"]), "actual": actual,
               "top25": main.top_n(p)}
        if actual is None:
            overdue = now.astimezone(config.IST) > config.result_datetime(market, d) + dt.timedelta(days=2)
            row["status"] = "no-result" if overdue else "pending"
        else:
            sc = _score(np.array(p["dist"]), None, None, actual, row["andar"], row["bahar"])
            row.update(status="hit" if sc["hit10"] else "miss", rank=sc["rank"], hit1=sc["hit1"],
                       andar_hit=sc["andar_hit"], bahar_hit=sc["bahar_hit"],
                       andar1_hit=sc["andar1_hit"], bahar1_hit=sc["bahar1_hit"], **main.grid_scores(row, actual))
            i = pe.st.pos.get((market, d))
            if i is not None and i >= START:
                row["why"] = _why(pe, i, sc, row["andar"], row["bahar"])
                lesson = _lesson(p, now_used)
                if lesson:
                    row["why"].append(lesson)
            if not p.get("late") and row["verified"]:
                scores.append(sc)
                counted.append(row)
        rows.append(row)
    digits = {key: {"hits": sum(1 for r in counted if r[key]), "n": len(counted), "chance": chance}
              for key, chance in (("andar_hit", DIGIT_K / 10), ("bahar_hit", DIGIT_K / 10),
                                  ("andar1_hit", 0.1), ("bahar1_hit", 0.1),
                                  ("grid_hit", main.GRID_N / 100), ("top25_hit", main.GRID_N / 100))}
    return {"summary": main._clean(summarize(scores)), "digits": digits, "rows": rows}


def _compare(pe: PatternEngine, rec: recency.Recency, positions: list[int], cache: dict) -> dict:
    """Pattern Engine vs engine 4.0 (recency) on exactly the same draws."""
    st = pe.st
    jp = j4 = ap = a4 = bp = b4 = 0
    for i in positions:
        y = int(st.vals[i])
        jp += rank_of(pe.jodi.P[i], y) <= 10
        a, b = y // 10, y % 10
        ap += a in _digits(pe.andar.P[i])
        bp += b in _digits(pe.bahar.P[i])
        if i not in cache:
            cache[i] = rec.dist(rec.pos[(st.markets[i], st.dates[i])])[0]
        p4 = cache[i]
        j4 += rank_of(p4, y) <= 10
        at, ab = andar_bahar(p4)
        a4 += a in _digits(at)
        b4 += b in _digits(ab)
    n = max(1, len(positions))
    return {"n": len(positions),
            "pattern": {"top10": main._r(jp / n), "andar5": main._r(ap / n), "bahar5": main._r(bp / n)},
            "engine40": {"top10": main._r(j4 / n), "andar5": main._r(a4 / n), "bahar5": main._r(b4 / n)},
            "p_pattern": main._r(binom_sf(int(jp), len(positions), 0.1), 6) if positions else None}


def _analyse(table, market, preds, now, lock_new, pe, rec, closed, cache):
    sd = SeriesData(table, market)
    meta = config.MARKETS[market]
    base = {"key": market, "name": meta["name"], "short": meta["short"], "result_time": meta["result_time"],
            "n_results": sd.n, "first": sd.dates[0].isoformat() if sd.n else None,
            "last": sd.dates[-1].isoformat() if sd.n else None}
    st = pe.st
    positions = [int(i) for i in st.by_market[market] if i >= START]
    if sd.n < main.MIN_DATA or len(positions) < 10:
        return {**base, "ready": False, "reason": f"Pattern Engine ko kam se kam {START} results (sab markets) chahiye."}, None
    target = main.next_target_date(market, sd, now)
    locked = next((p for p in preds if p["market"] == market and p["date"] == target.isoformat()), None)
    ready, waiting, deadline = main.lock_status(table, sd, target, now, closed)
    new = None
    if locked is None and lock_new and ready:
        new = locked = make_prediction(pe, sd, target, now)
    wait_info = None
    if locked is None:
        wait_info = {"date": target.isoformat(), "deadline": deadline.isoformat(timespec="minutes"),
                     "for": [config.MARKETS[o]["name"] for o, _ in waiting],
                     "for_times": [config.MARKETS[o]["result_time"] for o, _ in waiting],
                     "for_dates": [d.isoformat() for _, d in waiting]}
    scores, rows = [], []
    ranks, a_ranks, b_ranks = [], [], []
    for i in positions:
        y = int(st.vals[i])
        sc = _score(pe.jodi.P[i], pe.andar.P[i], pe.bahar.P[i], y)
        scores.append(sc)
        ranks.append(sc["rank"])
        a_ranks.append(rank_of(pe.andar.P[i], y // 10))
        b_ranks.append(rank_of(pe.bahar.P[i], y % 10))
    for i, sc in list(zip(positions, scores))[-200:][::-1]:
        andar, bahar = _digits(pe.andar.P[i]), _digits(pe.bahar.P[i])
        rows.append({"date": st.dates[i].isoformat(), "actual": int(st.vals[i]),
                     "top10": [int(v) for v in np.argsort(-pe.jodi.P[i], kind="stable")[:10]],
                     "andar": andar, "bahar": bahar, **{k: sc[k] for k in (
                         "rank", "hit10", "hit1", "andar_hit", "bahar_hit", "andar1_hit", "bahar1_hit")},
                     "why": _why(pe, i, sc, andar, bahar)})
    hits = np.array([s["hit10"] for s in scores])
    payload = {**base, "ready": True, "closed_days": main.closed_days(sd), "next": locked, "waiting": wait_info,
               "backtest": {**_summaries(scores), "rows": rows},
               "coverage": _coverage(np.array(ranks), np.array(a_ranks), np.array(b_ranks)),
               "decision": _decision(hits),
               "arrival": main._arrival(market, storage.load_result_rows()),
               "compare": _compare(pe, rec, positions, cache),
               "live": None}
    return payload, new


def _nothing_to_lock(table: dict, preds: list[dict], now: dt.datetime) -> bool:
    have = {(p["market"], p["date"]) for p in preds}
    closed = main.closed_map(table)
    for m in config.MARKET_KEYS:
        sd = SeriesData(table, m)
        if sd.n < main.MIN_DATA:
            continue
        target = main.next_target_date(m, sd, now)
        if (m, target.isoformat()) not in have and main.lock_status(table, sd, target, now, closed)[0]:
            return False
    return True


def cycle(now: dt.datetime | None = None, lock_new: bool = True) -> dict:
    """One run of the Pattern Engine on the results already in data/results.csv."""
    now = now or config.now_ist()
    table = storage.load_table()
    preds = load_predictions()
    thash = main.table_hash(table)
    out_path = pattern_dir() / "dashboard.json"
    prev = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    if (prev.get("data_hash") == thash and prev.get("code_hash") == CODE_HASH
            and _nothing_to_lock(table, preds, now)):
        for payload in prev.get("markets", {}).values():   # only times and statuses change
            for row in (payload.get("live") or {}).get("rows", []):
                if row.get("status") == "pending" and now.astimezone(config.IST) > config.result_datetime(
                        payload["key"], dt.date.fromisoformat(row["date"])) + dt.timedelta(days=2):
                    row["status"] = "no-result"
        prev["generated_at"] = now.astimezone(config.IST).isoformat(timespec="seconds")
        out_path.write_text(json.dumps(prev, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        return {"generated_at": prev["generated_at"], "fast_path": True, "new_predictions": []}
    pe = PatternEngine(table)
    rec = recency.Recency(table)
    closed = main.closed_map(table)
    now_used = pe.next(config.MARKET_KEYS[0], now.date() + dt.timedelta(days=1))["jodi_used"]
    markets, created, cache = {}, [], {}
    for m in config.MARKET_KEYS:
        payload, new = _analyse(table, m, preds, now, lock_new, pe, rec, closed, cache)
        if new is not None:
            append_prediction(new)
            preds.append(new)
            created.append(f"{m} {new['date']}")
        markets[m] = payload
    for m, payload in markets.items():
        if payload.get("ready"):
            payload["live"] = _live(m, preds, table, pe, now, now_used)
    scan = {"jodi": pe.jodi.scan(len(pe.st)), "andar": pe.andar.scan(len(pe.st)), "bahar": pe.bahar.scan(len(pe.st))}
    allpos = [i for i in range(START, len(pe.st))]
    dash = {
        "generated_at": now.astimezone(config.IST).isoformat(timespec="seconds"),
        "engine": ENGINE, "kind": "pattern", "data_hash": thash, "code_hash": CODE_HASH,
        "primary": config.PRIMARY_MARKET, "history_start": config.HISTORY_START.isoformat(),
        "settings": {"gate_t": GATE_T, "trail": TRAIL, "refit": REFIT, "start": START,
                     "jodi_patterns": len(JODI), "digit_patterns": len(DIGIT)},
        "patterns": scan,
        "compare_all": _compare(pe, rec, allpos, cache),
        "markets": markets,
        "results": main.results_table(table),
    }
    out_path.write_text(json.dumps(main._clean(dash), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (pattern_dir() / "prediction_history.csv").write_text(main.prediction_history(markets), encoding="utf-8")
    return {"generated_at": dash["generated_at"], "new_predictions": created}
