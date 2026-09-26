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
import hashlib
import json
import logging
import math
import threading

import numpy as np

from . import config, scraper, storage
from .engine import formulas, theorems
from .engine.base import SeriesData, andar_bahar
from .engine.ensemble import (Replay, digit_top, postmortem, predict_next, rank_of, replay, score,
                              summarize, top_list)

log = logging.getLogger("satta.service")
_lock = threading.Lock()
MIN_DATA = 10


DAY_KINDS = {
    "month_end": lambda d: (d + dt.timedelta(days=1)).day == 1,
    "month_start": lambda d: d.day == 1,
}


def closed_days(sd: SeriesData) -> list[str]:
    """Day kinds on which this market (almost) never has a result, learned from its history.

    e.g. Faridabad/Ghaziabad/Gali close on the last day of the month and
    Disawar's chart skips the 1st.
    """
    if sd.n < 30:
        return []
    have = set(sd.dates)
    days = [sd.dates[0] + dt.timedelta(days=k) for k in range((sd.dates[-1] - sd.dates[0]).days + 1)]
    out = []
    for kind, test in DAY_KINDS.items():
        cands = [d for d in days if test(d)]
        missing = sum(1 for d in cands if d not in have)
        if len(cands) >= 3 and missing >= 0.8 * len(cands):
            out.append(kind)
    return out


def next_target_date(market: str, sd: SeriesData, now: dt.datetime) -> dt.date:
    now = now.astimezone(config.IST)
    today = now.date()
    cand = sd.dates[-1] + dt.timedelta(days=1) if sd.n else today
    if cand < today:
        cand = today
    closed = [DAY_KINDS[k] for k in closed_days(sd)]
    # skip learned holidays, and days whose result time passed long ago without a result
    while (any(test(cand) for test in closed)
           or now > config.result_datetime(market, cand) + dt.timedelta(hours=6)):
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


def _coverage(rep: Replay) -> dict | None:
    """How the hit rate grows when more numbers are covered: tested vs random.

    For every list size N the walk-forward backtest says how often the real
    number was inside the engine's top-N. Random picking gives N/100.
    """
    if not rep.steps:
        return None
    ranks = np.array([rank_of(s.mix, s.actual) for s in rep.steps])
    a_ranks, b_ranks = [], []
    for s in rep.steps:
        at, ab = andar_bahar(s.mix)
        a_ranks.append(rank_of(at, s.actual // 10))
        b_ranks.append(rank_of(ab, s.actual % 10))
    a_ranks, b_ranks = np.array(a_ranks), np.array(b_ranks)
    last = ranks[-200:]
    return {
        "days": len(ranks),
        "jodi": [{"n": n, "tested": _r((ranks <= n).mean()), "tested_200": _r((last <= n).mean()),
                  "random": n / 100} for n in (1, 5, 10, 20, 30, 40, 50)],
        "andar": [{"k": k, "tested": _r((a_ranks <= k).mean()), "random": k / 10} for k in (1, 2, 3, 5)],
        "bahar": [{"k": k, "tested": _r((b_ranks <= k).mean()), "random": k / 10} for k in (1, 2, 3, 5)],
    }


def _progress(rep: Replay, window: int = 50) -> dict | None:
    """Did learning help? Learned weights vs the same experts with equal weights (no learning)."""
    steps = rep.steps
    if len(steps) < 10:
        return None
    learned = [score(s.mix, s.actual) for s in steps]
    equal = [score(s.equal, s.actual) for s in steps]
    best = int(np.argmin(rep.losses))
    best_ranks = np.array([s.ranks[best] for s in steps])
    w = min(window, len(steps))

    def rolling(scores):
        h = np.cumsum([1.0 if s["hit10"] else 0.0 for s in scores])
        return [_r((h[i] - (h[i - w] if i >= w else 0)) / min(i + 1, w)) for i in range(len(scores))]

    stride = max(1, len(steps) // 300)
    idx = list(range(w - 1, len(steps), stride))
    tuning = []
    if rep.meta is not None:
        tuning = sorted(({"eta": e, "alpha": a, "weight": _r(float(v))}
                         for (e, a), v in zip(rep.grid, rep.meta)), key=lambda r: -r["weight"])
    return {
        "window": w,
        "tuning": tuning,
        "merge": ({"linear": _r(float(rep.merge[0])), "geometric": _r(float(rep.merge[1]))}
                  if rep.merge is not None else None),
        "learned": _clean(summarize(learned)),
        "equal": _clean(summarize(equal)),
        "best_expert": {"label": rep.experts[best].label,
                        "hit10_rate": _r((best_ranks <= 10).mean()),
                        "mean_rank": _r(best_ranks.mean(), 2)},
        "series": {"dates": [steps[i].date.isoformat() for i in idx],
                   "Seekh kar (learned)": [rolling(learned)[i] for i in idx],
                   "Bina seekhe (equal weights)": [rolling(equal)[i] for i in idx],
                   "Random chance": [0.1 for _ in idx]},
    }


def self_break(table: dict, market: str, shuffles: int = 3, last: int = 150) -> dict | None:
    """'Khud ko todo' test: run the same engine on copies of the history whose order is shuffled.

    Shuffling keeps how often each number appears but destroys every time pattern. If the engine
    scores about the same on shuffled history as on the real one, it has found no real pattern.
    """
    sd = SeriesData(table, market)
    if sd.n < config.MIN_TRAIN + 40:
        return None
    start = max(config.MIN_TRAIN, sd.n - last)

    def run(s):
        rep = replay(s, start=start)
        return summarize([score(x.mix, x.actual) for x in rep.steps])

    real = run(sd)
    rng = np.random.default_rng(20260926)
    fakes = []
    for _ in range(shuffles):
        vals = sd.y.copy()
        rng.shuffle(vals)
        t2 = dict(table)
        t2[market] = dict(zip(sd.dates, (int(v) for v in vals)))
        fakes.append(run(SeriesData(t2, market)))
    fake10 = float(np.mean([f["hit10"]["rate"] for f in fakes]))
    fake_rank = float(np.mean([f["mean_rank"]["value"] for f in fakes]))
    edge = real["hit10"]["rate"] - fake10
    return _clean({
        "days": real["n"], "shuffles": shuffles,
        "real_hit10": real["hit10"]["rate"], "shuffled_hit10": fake10,
        "real_mean_rank": real["mean_rank"]["value"], "shuffled_mean_rank": fake_rank,
        "edge": edge,
        "verdict": ("Asli data par engine shuffled data se kaafi behtar hai — time-pattern mila"
                    if edge > 0.05 and real["hit10"]["p_value"] < 0.05 else
                    "Asli aur shuffled data par barabar — engine ko koi time-pattern nahi mila"),
    })


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

def table_hash(table: dict) -> str:
    items = sorted((m, d.isoformat(), v) for m, s in table.items() for d, v in s.items())
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()[:16]


def _analyse_market(table, market, preds, now, lock_new: bool, prev_selfbreak=None):
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
        "closed_days": closed_days(sd),
        "next": locked,
        "backtest": _backtest(rep),
        "coverage": _coverage(rep),
        "live": None,  # filled by caller once new predictions are appended
        "weights": _weights_history(rep),
        "progress": _progress(rep),
        # the shuffle test is slow, so it is reused until the results change
        "selfbreak": prev_selfbreak if prev_selfbreak is not None else self_break(table, market),
        "experts": _experts_table(rep),
        "theorems": _clean(theorems.findings(sd, rep)),
        "formulas": _clean(fr),
    }
    return payload, (new, rep)


def _nothing_to_lock(table: dict, preds: list[dict], now: dt.datetime) -> bool:
    """True when every market's next draw already has a locked prediction."""
    have = {(p["market"], p["date"]) for p in preds}
    for m in config.MARKET_KEYS:
        sd = SeriesData(table, m)
        if sd.n >= MIN_DATA and (m, next_target_date(m, sd, now).isoformat()) not in have:
            return False
    return True


def _refresh_only(dash: dict, sync_info, now: dt.datetime) -> dict:
    """Fast path when no result changed: keep all analysis, update times and statuses."""
    for m, payload in dash.get("markets", {}).items():
        for row in (payload.get("live") or {}).get("rows", []):
            if row.get("status") == "pending":
                d = dt.date.fromisoformat(row["date"])
                if now.astimezone(config.IST) > config.result_datetime(m, d) + dt.timedelta(days=2):
                    row["status"] = "no-result"
    dash["generated_at"] = now.astimezone(config.IST).isoformat(timespec="seconds")
    if sync_info is not None:
        dash["sync"] = sync_info
    (config.DATA_DIR / "dashboard.json").write_text(
        json.dumps(_clean(dash), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {"generated_at": dash["generated_at"], "new_predictions": [], "fast_path": True,
            "results": sum(m.get("n_results", 0) for m in dash.get("markets", {}).values()),
            "sync": {k: v for k, v in (sync_info or {}).items() if k != "conflicts"}}


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
        thash = table_hash(table)
        prev = load_dashboard() or {}
        same = prev.get("engine") == config.ENGINE_VERSION and prev.get("data_hash") == thash
        if same and _nothing_to_lock(table, preds, now):
            return _refresh_only(prev, sync_info, now)
        markets, created = {}, []
        for m in config.MARKET_KEYS:
            old = (prev.get("markets") or {}).get(m) or {}
            payload, extra = _analyse_market(table, m, preds, now, lock_new,
                                             old.get("selfbreak") if same else None)
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
            "data_hash": thash,
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
