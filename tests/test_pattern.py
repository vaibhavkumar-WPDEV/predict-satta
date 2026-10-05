import datetime as dt
import json
import random

import numpy as np

from satta import config, storage
from satta.pattern import engine as E
from satta.pattern import service as P

from .conftest import synthetic_rows, write_results


def at(y, m, d, hh, mm=0):
    return dt.datetime(y, m, d, hh, mm, tzinfo=config.IST)


def planted_table(days=160, seed=3):
    """Random results, except Gali repeats that evening's Ghaziabad on 60 % of days."""
    rnd = random.Random(seed)
    table = {m: {} for m in config.MARKET_KEYS}
    for k in range(days):
        d = dt.date(2025, 1, 1) + dt.timedelta(days=k)
        for m in ("disawar", "faridabad", "ghaziabad"):
            table[m][d] = rnd.randint(0, 99)
        table["gali"][d] = table["ghaziabad"][d] if rnd.random() < 0.6 else rnd.randint(0, 99)
    return table


def test_gate_finds_a_planted_pattern_and_predicts_with_it():
    pe = E.PatternEngine(planted_table())
    st = pe.st
    gali = [i for i in st.by_market["gali"] if i >= E.START]
    hits = np.mean([st.vals[i] in np.argsort(-pe.jodi.P[i], kind="stable")[:10] for i in gali])
    assert hits > 0.4                                   # random would be 0.10
    used = {r["name"] for r in pe.jodi.scan(len(st)) if r["used"]}
    assert "sameday" in used


def test_no_fake_edge_on_random_results():
    rnd = random.Random(11)
    table = {m: {dt.date(2025, 1, 1) + dt.timedelta(days=k): rnd.randint(0, 99) for k in range(220)}
             for m in config.MARKET_KEYS}
    pe = E.PatternEngine(table)
    st = pe.st
    idx = range(E.START, len(st))
    hits = np.mean([st.vals[i] in np.argsort(-pe.jodi.P[i], kind="stable")[:10] for i in idx])
    assert 0.05 < hits < 0.16                          # about the random 10 %


def test_features_never_look_ahead():
    table = planted_table(days=120)
    st = E.Stream(table)
    i = 300
    m, d = st.markets[i], st.dates[i]
    before = E.jodi_features(st, i, m, d)
    later = {k: dict(v) for k, v in table.items()}
    for k in later:                                     # change every result from draw i on
        for day in list(later[k]):
            if (day, st.slot[k]) >= st.keys[i]:
                later[k][day] = (later[k][day] + 17) % 100
    st2 = E.Stream(later)
    assert np.array_equal(before, E.jodi_features(st2, i, m, d))
    assert np.array_equal(E.digit_features(st, i, m, d, "a"), E.digit_features(st2, i, m, d, "a"))


def test_pattern_cycle_uses_its_own_files_and_scores_later(data_dir):
    rows = synthetic_rows(140)                          # 2026-01-01 .. 2026-05-20, 560 draws
    write_results(data_dir / "results.csv", rows)
    main_preds = data_dir / "predictions.jsonl"
    out = P.cycle(now=at(2026, 5, 21, 1))               # Disawar 21 May: all of last night is in
    assert out["new_predictions"] == ["disawar 2026-05-21"]
    assert not main_preds.exists()                      # engine 4.0's file is never touched
    preds = P.load_predictions()
    p = preds[0]
    assert P.verify(p) and len(p["top10"]) == 10 and len(p["andar"]) == len(p["bahar"]) == 5
    p2 = dict(p, andar=[[9, 0.5]] + p["andar"][1:])
    assert not P.verify(p2)                              # digits are inside the hash
    # result comes: next run scores it from the locked lists
    actual = p["top10"][0][0]
    rows.append({"market": "disawar", "date": "2026-05-21", "value": actual, "source": "t", "fetched_at": ""})
    write_results(data_dir / "results.csv", rows)
    P.cycle(now=at(2026, 5, 21, 6))
    dash = json.loads((data_dir / "pattern" / "dashboard.json").read_text())
    row = [r for r in dash["markets"]["disawar"]["live"]["rows"] if r["date"] == "2026-05-21"][0]
    assert row["status"] == "hit" and row["rank"] == 1 and row["verified"]
    assert row["top25_hit"] and row["grid_hit"] == (actual // 10 in row["andar"] and actual % 10 in row["bahar"])
    assert dash["kind"] == "pattern" and dash["patterns"]["jodi"] and dash["compare_all"]["n"] > 0
    assert (data_dir / "pattern" / "prediction_history.csv").read_text().startswith("date,market")
    # nothing new and nothing to lock: the fast path only refreshes times
    assert P.cycle(now=at(2026, 5, 21, 7)).get("fast_path") is True
    assert storage.load_predictions() == []
