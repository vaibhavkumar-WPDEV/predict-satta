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
        # Disawar is declared before Faridabad on the same day, so its cutoff-day
        # result may be used; everything from Faridabad's cutoff draw on may not.
        first_bad = cutoff + dt.timedelta(days=1) if m == "disawar" else cutoff
        for d in t2[m]:
            if d >= first_bad:
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
    assert best.startswith("formula") or best in ("cross_market", "transforms", "genetic_formula")


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


def test_ctw_learns_a_repeating_pattern():
    from satta.engine.advanced import CTW

    c = CTW(depth=3)
    seq = [1, 2, 3, 1, 2, 7] * 40
    for x in seq:
        c.update(x)
    p = c.predict()  # history ends ... 2, 7 -> next is 1
    assert abs(p.sum() - 1) < 1e-9
    assert p.argmax() == 1 and p[1] > 0.85
    # after "1, 2" the next symbol depends on deeper context (3 or 7)
    c.update(1)
    c.update(2)
    assert c.predict().argmax() == 3


def test_neural_net_learns_cross_market_digit_rule():
    rows = synthetic_rows(300, seed=9)
    tab = table_from(rows)
    for d in sorted(tab["faridabad"]):
        prev = tab["gali"].get(d - dt.timedelta(days=1))
        if prev is not None:  # andar(FRBD) = bahar(GALI kal), bahar random
            tab["faridabad"][d] = (prev % 10) * 10 + tab["faridabad"][d] % 10
    sd = SeriesData(tab, "faridabad")
    from satta.engine.advanced import NeuralNet

    nn = NeuralNet()
    hits = 0
    for i in range(200, sd.n):
        p = nn.predict(sd.context(i))
        hits += int(p.reshape(10, 10).sum(1).argmax() == sd.y[i] // 10)
    assert hits / (sd.n - 200) > 0.8


def test_same_day_earlier_market_is_a_feature():
    tab = table_from(synthetic_rows(120, seed=4))
    sd = SeriesData(tab, "ghaziabad")
    assert sd.earlier == ["disawar", "faridabad"]
    d = sd.dates[50]
    assert sd.F["faridabad@0"][50] == tab["faridabad"][d]
    assert sd.F["disawar@0"][50] == tab["disawar"][d]
    assert "gali@0" not in sd.F  # Gali is declared after Ghaziabad


def test_same_day_pattern_is_learned():
    """Faridabad = palti of the same morning's Disawar on 40 % of days."""
    import random

    rng = random.Random(8)
    tab = table_from(synthetic_rows(260, seed=8))
    for d in tab["faridabad"]:
        if d in tab["disawar"] and rng.random() < 0.4:
            x = tab["disawar"][d]
            tab["faridabad"][d] = (x % 10) * 10 + x // 10
    rep = replay(SeriesData(tab, "faridabad"))
    s = summarize([score(x.mix, x.actual) for x in rep.steps[-100:]])
    assert s["hit1"]["rate"] > 0.25


def test_miss_correction_learns_a_systematic_shift():
    """An expert that is always one below the truth: the correction layer learns '+1'."""
    from satta.engine.experts import Expert

    class OneBelow(Expert):
        name = "one_below"

        def predict(self, ctx):
            p = np.full(100, 0.1 / 99)
            p[(int(ctx.sd.y[ctx.i]) - 1) % 100] = 0.9  # test-only oracle, shifted by -1
            return p

    sd = SeriesData(table_from(synthetic_rows(120, seed=6)), "faridabad")
    rep = replay(sd, experts=[OneBelow()])
    from satta.engine.ensemble import SHIFTS

    assert list(SHIFTS)[int(np.argmax(rep.correction))] == "+1"
    s = summarize([score(x.mix, x.actual) for x in rep.steps[-40:]])
    assert s["hit1"]["rate"] == 1.0


def test_top10_selector_uses_the_list_that_hits():
    """An expert whose list always holds the result but whose probabilities are nearly flat:
    log-loss hardly rewards it, the Top-10 selector does."""
    from satta.engine.experts import Expert, Uniform

    class NearlyFlat(Expert):
        name = "nearly_flat"

        def predict(self, ctx):
            p = np.full(100, 1.0)
            y = int(ctx.sd.y[ctx.i])  # test-only oracle: the result is always in its top 10
            for v in range(y, y + 10):
                p[v % 100] = 1.02
            return p / p.sum()

    sd = SeriesData(table_from(synthetic_rows(200, seed=12)), "faridabad")
    rep = replay(sd, experts=[Uniform(), NearlyFlat()])
    s = summarize([score(x.mix, x.actual) for x in rep.steps[-80:]])
    assert s["hit10"]["rate"] == 1.0
    # the flat-but-right expert scores far above the uniform one; on a tie with
    # the ensemble (which also learned it) the ensemble's list is kept
    assert rep.selector[1] > rep.selector[0] + 1
    assert rep.chosen[-1] in (1, 2)


def test_formula_report_finds_planted_formula():
    sd = SeriesData(table_from(synthetic_rows(200, seed=5, planted=True)), "faridabad")
    rep = formulas.report(sd, sd.features_for(dt.date(2026, 7, 20), sd.n))
    top = rep["kinds"]["jodi"]["top"][0]
    assert top["formula"] == "FRBD = (GALI(kal) + 37) mod 100"
    assert top["significant"]


def test_theorems_flag_cross_market_only_when_real():
    sd = SeriesData(table_from(synthetic_rows(260, seed=5, planted=True)), "faridabad")
    f = {t["id"]: t for t in theorems.findings(sd)}
    assert f["T8GAA"]["pattern"] is True
    assert f["T1"]["pattern"] is False
