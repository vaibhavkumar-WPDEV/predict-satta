import datetime as dt

import numpy as np

from satta.engine import formulas, theorems
from satta.engine.base import SeriesData
from satta.engine.ensemble import replay, score, summarize
from satta.engine.experts import default_experts

from .conftest import synthetic_rows


def table_from(rows):
    t = {m: {} for m in ("disawar", "faridabad", "ghaziabad", "gali")}
    for r in rows:
        t[r["market"]][dt.date.fromisoformat(r["date"])] = int(r["value"])
    return t


def test_every_expert_returns_a_distribution():
    sd = SeriesData(table_from(synthetic_rows(120)), "faridabad")
    for i in (0, 5, 40, sd.n):
        ctx = sd.context(i, dt.date(2026, 6, 1) if i == sd.n else None)
        for e in default_experts():
            p = e.predict(ctx)
            assert p.shape == (100,), e.name
            assert np.all(p >= 0), e.name
            assert abs(p.sum() - 1) < 1e-9, e.name


def test_no_future_leak():
    """Changing results after day i must not change the prediction for day i."""
    rows = synthetic_rows(110, seed=3)
    t1 = table_from(rows)
    t2 = table_from(rows)
    cutoff = dt.date(2026, 1, 1) + dt.timedelta(days=80)
    for m in t2:
        for d in t2[m]:
            if d >= cutoff:
                t2[m][d] = (t2[m][d] + 50) % 100
    r1 = replay(SeriesData(t1, "faridabad"))
    r2 = replay(SeriesData(t2, "faridabad"))
    s1 = {s.date: s.mix for s in r1.steps}
    s2 = {s.date: s.mix for s in r2.steps}
    for d in s1:
        if d < cutoff:
            assert np.allclose(s1[d], s2[d]), d
        # the day of the cutoff itself only uses earlier data too
    assert np.allclose(s1[cutoff], s2[cutoff])


def test_random_data_is_not_beaten():
    sd = SeriesData(table_from(synthetic_rows(200, seed=11)), "faridabad")
    rep = replay(sd)
    s = summarize([score(x.mix, x.actual) for x in rep.steps])
    assert s["hit10"]["p_value"] > 0.001  # no fake "edge" on pure noise


def test_planted_pattern_is_learned():
    sd = SeriesData(table_from(synthetic_rows(200, seed=5, planted=True)), "faridabad")
    rep = replay(sd)
    s = summarize([score(x.mix, x.actual) for x in rep.steps[-100:]])
    assert s["hit1"]["rate"] > 0.25
    assert s["hit1"]["p_value"] < 1e-6
    best = rep.experts[int(np.argmax(rep.weights))].name
    assert best.startswith("formula") or best in ("cross_market", "transforms")


def test_aryabhata_cracks_a_linear_congruential_generator():
    rows = synthetic_rows(160, seed=2)
    x = 17
    for r in rows:
        if r["market"] == "faridabad":
            x = (37 * x + 11) % 100
            r["value"] = x
    sd = SeriesData(table_from(rows), "faridabad")
    rep = replay(sd)
    best = rep.experts[int(np.argmax(rep.weights))].name
    assert best == "aryabhata_lcg"
    s = summarize([score(x.mix, x.actual) for x in rep.steps[-60:]])
    assert s["hit1"]["rate"] > 0.9
    fr = formulas.report(sd, sd.features_for(dt.date(2026, 6, 20), sd.n))
    assert fr["kinds"]["aryabhata"]["top"][0]["formula"] == "FRBD = (37·FRBD(pichla) + 11) mod 100"


def test_formula_report_finds_planted_formula():
    sd = SeriesData(table_from(synthetic_rows(200, seed=5, planted=True)), "faridabad")
    rep = formulas.report(sd, sd.features_for(dt.date(2026, 7, 20), sd.n))
    top = rep["kinds"]["jodi"]["top"][0]
    assert top["formula"] == "FRBD = (GALI(kal) + 37) mod 100"
    assert top["significant"]


def test_theorems_flag_cross_market_only_when_real():
    sd = SeriesData(table_from(synthetic_rows(260, seed=5, planted=True)), "faridabad")
    f = {t["id"]: t for t in theorems.findings(sd)}
    assert f["T8GA"]["pattern"] is True
    assert f["T1"]["pattern"] is False
