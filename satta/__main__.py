"""Command line: python -m satta <command>

  serve      web dashboard + backend + automatic scheduler (http://localhost:8000)
  cycle      one full run: fetch -> check -> learn -> predict -> lock
  sync       only download results (use --full to re-download from Jan 2026)
  predict    show the locked next prediction of every market
  backtest   walk-forward test, e.g. --market faridabad --days 7
  table      result table of all markets
  theorems   statistical findings for a market
  formulas   discovered formulas and their holdout test
  import     fallback: import a CSV (date,market,value)
  verify     check the SHA-256 proof of every locked prediction
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import config, scraper, service, storage


def _pct(x):
    return f"{x * 100:.1f}%"


def cmd_serve(a):
    import uvicorn

    print(f"Dashboard: http://localhost:{a.port}   (Ctrl+C se band karo)")
    uvicorn.run("satta.api:app", host=a.host, port=a.port, log_level="info")


def cmd_cycle(a):
    res = service.cycle(fetch=not a.no_fetch)
    print(f"Cycle done {res['generated_at']}: {res['results']} results, "
          f"new predictions: {', '.join(res['new_predictions']) or 'none'}")
    if res.get("sync"):
        print("sync:", res["sync"])
    cmd_predict(a)


def cmd_sync(a):
    print(scraper.sync(full=a.full))


def cmd_predict(a):
    dash = service.load_dashboard() or (service.cycle(fetch=False) and service.load_dashboard())
    for key, m in dash["markets"].items():
        if not m.get("ready"):
            print(f"\n{m['name']}: {m.get('reason')}")
            continue
        p = m["next"]
        if not p:
            w = m.get("waiting") or {}
            print(f"\n== {m['name']} ({m['short']}) — {w.get('date')}: prediction "
                  f"{', '.join(w.get('for', [])) + ' ke aaj ke result ke baad' if w.get('for') else 'jaldi'} "
                  f"lock hogi (deadline {w.get('deadline')})")
            continue
        print(f"\n== {m['name']} ({m['short']}) — {p['date']}  result ~{m['result_time']} IST ==")
        print("  Top-10 jodi :", "  ".join(f"{v:02d}({_pct(pr)})" for v, pr in p["top10"]))
        print("  Andar top-3 :", ", ".join(f"{d}({_pct(pr)})" for d, pr in p["andar"]))
        print("  Bahar top-3 :", ", ".join(f"{d}({_pct(pr)})" for d, pr in p["bahar"]))
        for f in p.get("formulas", []):
            print("  Formula     :", f)
        print(f"  Locked at   : {p['created_at']}{'  (LATE: result ke baad bani)' if p['late'] else ''}")
        print(f"  SHA-256     : {p['hash']}")


def cmd_backtest(a):
    rows, summary = service.backtest(a.market, a.days)
    print(f"Walk-forward backtest: {a.market}, last {a.days} draws (har din sirf pichla data use hua)\n")
    for r in rows:
        mark = "HIT " if r["hit10"] else "MISS"
        print(f"{r['date']}  actual {r['actual']:02d}  rank {r['rank']:>3}  {mark}  "
              f"top10: {' '.join(f'{v:02d}' for v in r['top10'])}")
        for line in r["why"]:
            print("            ", line)
    print()
    _print_summary(summary)


def _print_summary(s):
    if not s.get("n"):
        print("No data")
        return
    for k, name in (("hit1", "Exact (top-1)"), ("hit5", "Top-5"), ("hit10", "Top-10"),
                    ("andar_hit", "Andar top-3"), ("bahar_hit", "Bahar top-3")):
        v = s[k]
        print(f"{name:14} {v['hits']:>4}/{s['n']}  = {_pct(v['rate']):>6}   random: {_pct(v['chance']):>6}   "
              f"p-value {v['p_value']:.3f}")
    print(f"Mean rank      {s['mean_rank']['value']:.1f}   random: 50.5")


def cmd_table(a):
    rows = service.results_table(storage.load_table())
    since = a.since or config.HISTORY_START.isoformat()
    print("date        " + "  ".join(f"{config.MARKETS[m]['short']:>4}" for m in config.MARKET_KEYS))
    for r in reversed(rows):
        if r["date"] < since:
            continue
        print(r["date"] + "  " + "  ".join(f"{r[m]:>4}" if r[m] is None else f"  {r[m]:02d}"
                                           for m in config.MARKET_KEYS).replace("None", "  XX"))


def _market_payload(market):
    dash = service.load_dashboard() or (service.cycle(fetch=False) and service.load_dashboard())
    return dash["markets"][market]


def cmd_theorems(a):
    m = _market_payload(a.market)
    for t in m.get("theorems", []):
        p = f"p={t['p_value']:.4f}" if t["p_value"] is not None else ""
        print(f"[{t['id']}] {t['title']}\n     {t['statement']}\n     {t['stat']} {p}\n     => {t['verdict']}")


def cmd_formulas(a):
    fr = _market_payload(a.market).get("formulas", {})
    if not fr.get("ready"):
        print(fr.get("reason"))
        return
    print(f"Discovery: pehle {fr['train_days']} din par, test: aakhri {fr['test_days']} din (unseen)\n")
    for kind, k in fr["kinds"].items():
        print(f"--- {kind}: {k['tested']} formule try kiye, random chance {_pct(k['chance_rate'])}")
        for r in k["top"]:
            print(f"  {r['formula']}\n      train {r['train_hits']}/{r['train_n']} ({_pct(r['train_rate'])})  "
                  f"test {r['test_hits']}/{r['test_n']} ({_pct(r['test_rate'])})  p={r['p_value']:.3f}  "
                  f"{'SIGNIFICANT' if r['significant'] else 'chance jaisa'}")
        for r in k["next"]:
            print(f"  NEXT: {r['formula']} → {r['value']}")


def cmd_import(a):
    print(f"{scraper.import_csv(a.path)} rows imported/updated")


def cmd_verify(a):
    preds = storage.load_predictions()
    bad = [p for p in preds if not storage.verify_prediction(p)]
    print(f"{len(preds)} locked predictions, {len(preds) - len(bad)} verified, {len(bad)} tampered")
    for p in bad:
        print("  TAMPERED:", p["market"], p["date"])


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(prog="python -m satta", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("cycle")
    s.add_argument("--no-fetch", action="store_true")
    s.set_defaults(fn=cmd_cycle)
    s = sub.add_parser("sync")
    s.add_argument("--full", action="store_true")
    s.set_defaults(fn=cmd_sync)
    sub.add_parser("predict").set_defaults(fn=cmd_predict)
    s = sub.add_parser("backtest")
    s.add_argument("--market", default=config.PRIMARY_MARKET, choices=config.MARKET_KEYS)
    s.add_argument("--days", type=int, default=7)
    s.set_defaults(fn=cmd_backtest)
    s = sub.add_parser("table")
    s.add_argument("--since")
    s.set_defaults(fn=cmd_table)
    for name, fn in (("theorems", cmd_theorems), ("formulas", cmd_formulas)):
        s = sub.add_parser(name)
        s.add_argument("--market", default=config.PRIMARY_MARKET, choices=config.MARKET_KEYS)
        s.set_defaults(fn=fn)
    s = sub.add_parser("import")
    s.add_argument("path")
    s.set_defaults(fn=cmd_import)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
