"""The daily loop: fetch -> evaluate -> learn -> predict -> lock -> publish.

cycle() is idempotent and can run as often as you like (the server runs it
every 15 minutes, GitHub Actions every hour):

1. download new results (scraper.sync)
2. replay the whole history walk-forward; the ensemble re-learns its weights
   from every real result (this is the self-correction)
3. for each market, if the next draw has no locked prediction yet, create one,
   hash it (SHA-256) and append it to data/predictions.jsonl. A locked
   prediction is never changed again, so it is proof it was made before the
   result.
4. score every locked prediction whose result is now known, explain the
   hit/miss, and write data/dashboard.json for the frontend.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import threading

import numpy as np

from . import config, scraper, storage
from .engine import formulas, theorems
from .engine.base import SeriesData, andar_bahar
from .engine.ensemble import (Replay, digit_top, postmortem, predict_next, replay, score,
                              summarize, top_list)

log = logging.getLogger("satta.service")
_lock = threading.Lock()
MIN_DATA = 10


def next_target_date(market: str, sd: SeriesData, now: dt.datetime) -> dt.date:
    now = now.astimezone(config.IST)
    today = now.date()
    cand = sd.dates[-1] + dt.timedelta(days=1) if sd.n else today
    if cand < today:
        cand = today
    # if the result time passed long ago and still nothing: holiday -> next day
    while now > config.result_datetime(market, cand) + dt.timedelta(hours=6):
        cand += dt.timedelta(days=1)
    return cand


def make_prediction(sd: SeriesData, rep: Replay, date: dt.date, now: dt.datetime,
                    formula_report: dict | None = None) -> dict:
    mix, _ = predict_next(sd, rep, date)
    at, ab = andar_bahar(mix)
    rt = config.result_datetime(sd.market, date)
    order = np.argsort(-rep.weights)[:6]
    pred = {
        "market": sd.market,
        "date": date.isoformat(),
        "created_at": now.astimezone(config.IST).isoformat(timespec="seconds"),
        "result_time": rt.isoformat(timespec="minutes"),
        "late": now > rt,
        "engine": config.ENGINE_VERSION,
        "trained_on": sd.n,
        "last_result": sd.dates[-1].isoformat() if sd.n else None,
        "top10": top_list(mix, 10),
        "andar": digit_top(at),
        "bahar": digit_top(ab),
        "dist": [round(float(x), 5) for x in mix],
        "experts": [[rep.experts[i].label, round(float(rep.weights[i]), 4)] for i in order],
    }
    if formula_report and formula_report.get("ready"):
        pred["formulas"] = [f"{f['formula']} → {f['value']:02d}"
                            for f in formula_report["kinds"]["jodi"]["next"][:3]]
    pred["hash"] = storage.prediction_hash(pred)
    return pred


# ------------------------------------------------------------ payload

def _r(x, nd=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def _weights_history(rep: Replay, keep: int = 8) -> dict:
    if not rep.steps:
        return {"dates": [], "series": {}}
    W = np.stack([s.new_weights for s in rep.steps])
    idx = list(np.argsort(-rep.weights)[:keep])
    uni = next((i for i, e in enumerate(rep.experts) if e.name == "uniform"), None)
    if uni is not None and uni not in idx:
        idx.append(uni)
    stride = max(1, len(rep.steps) // 300)
    rows = range(0, len(rep.steps), stride)
    return {
        "dates": [rep.steps[r].date.isoformat() for r in rows],
        "series": {rep.experts[i].label: [_r(W[r, i]) for r in rows] for i in idx},
    }


def _experts_table(rep: Replay) -> list[dict]:
    out = []
    n = len(rep.steps)
    for i, e in enumerate(rep.experts):
        ranks = np.array([s.ranks[i] for s in rep.steps]) if n else np.array([])
        out.append({
            "name": e.name, "label": e.label, "theory": e.theory,
            "weight": _r(rep.weights[i]),
            "mean_rank": _r(ranks.mean(), 2) if n else None,
            "hit10_rate": _r((ranks <= 10).mean()) if n else None,
            "avg_logloss": _r(rep.losses[i] / n) if n else None,
        })
    out.sort(key=lambda r: -(r["weight"] or 0))
    return out


def _backtest(rep: Replay, rows: int = 200) -> dict:
    scores = [score(s.mix, s.actual) for s in rep.steps]
    table = []
    for s, sc in list(zip(rep.steps, scores))[-rows:][::-1]:
        table.append({
            "date": s.date.isoformat(), "actual": s.actual,
            "top10": [v for v, _ in top_list(s.mix)],
            "rank": sc["rank"], "hit10": sc["hit10"], "hit1": sc["hit1"],
            "andar_hit": sc["andar_hit"], "bahar_hit": sc["bahar_hit"],
            "why": postmortem(s, rep.experts, sc)["lines"],
        })
    return {
        "all": _clean(summarize(scores)),
        "last200": _clean(summarize(scores[-200:])),
        "last100": _clean(summarize(scores[-100:])),
        "last30": _clean(summarize(scores[-30:])),
        "last7": _clean(summarize(scores[-7:])),
        "rows": table,
    }


def _live(market: str, preds: list[dict], table: dict, rep: Replay, now: dt.datetime) -> dict:
    steps = {s.date.isoformat(): s for s in rep.steps}
    rows, scores = [], []
    for p in sorted((p for p in preds if p["market"] == market), key=lambda p: p["date"], reverse=True):
        d = dt.date.fromisoformat(p["date"])
        actual = table[market].get(d)
        row = {"date": p["date"], "created_at": p["created_at"], "late": p.get("late", False),
               "hash": p["hash"], "verified": storage.verify_prediction(p),
               "top10": [v for v, _ in p["top10"]], "andar": [d_ for d_, _ in p["andar"]],
               "bahar": [d_ for d_, _ in p["bahar"]], "actual": actual}
        if actual is None:
            overdue = now.astimezone(config.IST) > config.result_datetime(market, d) + dt.timedelta(days=2)
            row["status"] = "no-result" if overdue else "pending"
        else:
            sc = score(np.array(p["dist"]), actual)
            row.update(status="hit" if sc["hit10"] else "miss", rank=sc["rank"], hit1=sc["hit1"],
                       andar_hit=sc["andar_hit"], bahar_hit=sc["bahar_hit"])
            step = steps.get(p["date"])
            if step is not None:
                row["why"] = postmortem(step, rep.experts, sc)["lines"]
            if not p.get("late") and row["verified"]:
                scores.append(sc)
        rows.append(row)
    return {"summary": _clean(summarize(scores)), "rows": rows}


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _r(obj, 6)
    return obj


def results_table(table: dict) -> list[dict]:
    dates = sorted({d for m in table.values() for d in m})
    return [{"date": d.isoformat(), **{m: table[m].get(d) for m in config.MARKET_KEYS}}
            for d in reversed(dates)]


# -------------------------------------------------------------- cycle

def _analyse_market(table, market, preds, now, lock_new: bool):
    sd = SeriesData(table, market)
    meta = config.MARKETS[market]
    base = {"key": market, "name": meta["name"], "short": meta["short"],
            "result_time": meta["result_time"], "n_results": sd.n,
            "first": sd.dates[0].isoformat() if sd.n else None,
            "last": sd.dates[-1].isoformat() if sd.n else None}
    if sd.n < MIN_DATA:
        return {**base, "ready": False,
                "reason": f"Is market ke sirf {sd.n} result hain, kam se kam {MIN_DATA} chahiye."}, None
    rep = replay(sd, start=min(config.MIN_TRAIN, sd.n))
    target = next_target_date(market, sd, now)
    fr = formulas.report(sd, sd.features_for(target, sd.n))
    new = None
    locked = next((p for p in preds if p["market"] == market and p["date"] == target.isoformat()), None)
    if locked is None and lock_new:
        new = make_prediction(sd, rep, target, now, fr)
        locked = new
    payload = {
        **base, "ready": True,
        "next": locked,
        "backtest": _backtest(rep),
        "live": None,  # filled by caller once new predictions are appended
        "weights": _weights_history(rep),
        "experts": _experts_table(rep),
        "theorems": _clean(theorems.findings(sd, rep)),
        "formulas": _clean(fr),
    }
    return payload, (new, rep)


def cycle(fetch: bool = True, now: dt.datetime | None = None, lock_new: bool = True) -> dict:
    with _lock:
        now = now or config.now_ist()
        sync_info = None
        if fetch:
            try:
                sync_info = scraper.sync(now)
            except Exception as exc:  # network problems must not stop predictions
                log.exception("sync failed")
                sync_info = {"error": str(exc)}
            (storage.cache_dir() / "sync.json").write_text(json.dumps(sync_info, default=str, indent=1))
        elif (storage.cache_dir() / "sync.json").exists():
            sync_info = json.loads((storage.cache_dir() / "sync.json").read_text())

        table = storage.load_table()
        preds = storage.load_predictions()
        markets, created = {}, []
        for m in config.MARKET_KEYS:
            payload, extra = _analyse_market(table, m, preds, now, lock_new)
            if extra and extra[0] is not None:
                storage.append_prediction(extra[0])
                preds.append(extra[0])
                created.append(f"{m} {extra[0]['date']}")
            if extra:
                payload["live"] = _live(m, preds, table, extra[1], now)
            markets[m] = payload

        dash = {
            "generated_at": now.astimezone(config.IST).isoformat(timespec="seconds"),
            "engine": config.ENGINE_VERSION,
            "primary": config.PRIMARY_MARKET,
            "history_start": config.HISTORY_START.isoformat(),
            "sync": sync_info,
            "markets": markets,
            "results": results_table(table),
        }
        (config.DATA_DIR / "dashboard.json").write_text(
            json.dumps(_clean(dash), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        return {"generated_at": dash["generated_at"], "new_predictions": created,
                "results": sum(len(v) for v in table.values()),
                "sync": {k: v for k, v in (sync_info or {}).items() if k != "conflicts"}}


def load_dashboard() -> dict | None:
    path = config.DATA_DIR / "dashboard.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def backtest(market: str, days: int = 7) -> tuple[list[dict], dict]:
    """Walk-forward test of the last `days` draws (what the CLI prints)."""
    sd = SeriesData(storage.load_table(), market)
    if sd.n < MIN_DATA + days:
        raise ValueError(f"{market}: sirf {sd.n} result hain")
    rep = replay(sd, start=min(config.MIN_TRAIN, sd.n - days))
    rows = []
    for s in rep.steps[-days:]:
        sc = score(s.mix, s.actual)
        rows.append({"date": s.date.isoformat(), "actual": s.actual,
                     "top10": [v for v, _ in top_list(s.mix)], **sc,
                     "why": postmortem(s, rep.experts, sc)["lines"]})
    return rows, summarize([score(s.mix, s.actual) for s in rep.steps[-days:]])
