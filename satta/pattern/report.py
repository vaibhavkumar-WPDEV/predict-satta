"""ENGINE 5 analysis report for one locked prediction.

Everything in the report comes from the data, and nothing in it changes the locked list:
the ranking is the locked one; the evidence is each pattern's push on a number (weight x
how far the number's pattern value is from the average). "Estimated probability" is
measured, not claimed: how often the real number had that rank in the walk-forward test.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from ..engine.stats import binom_sf
from .engine import LABELS, START, Transitions, _zs, probs

BUCKETS = [(1, 5), (6, 10), (11, 25), (26, 50), (51, 100)]

# OLD METHOD -> FAILURE -> NEW DISCOVERY -> CHANGE -> TEST RESULT (same 7,739 draws, walk-forward)
UPGRADES = [
    {"version": "5.0", "date": "2026-10-05",
     "old": "Pattern Engine P-1.0: recency, gap, balancing, digit patterns (19 patterns, gate |t| > 2)",
     "failure": "Ghaziabad 5 Oct = 78 ko rank 87 diya; 'pichle number ke baad kya aata hai' kabhi check nahi hua tha",
     "discovery": "Koi market X dene ke baad, X ke baad pehle aa chuke numbers kam dohrata hai: Disawar 6.8% "
                  "(umeed 10.7%, z = -5.3), Gali 7.5% (10.5%, z = -4.3), Faridabad / Ghaziabad z = -2.5; "
                  "shuffled data par yeh gayab (z ~ 0), dono aadhe 5 saal me same",
     "change": "Jodi model me 4 transition patterns (isi market ka pichla -> agla, koi bhi market ka pichla -> agla, "
               "digit -> digit); gate aur weights data se",
     "test": "Top-10 11.95% -> 12.49% (pichle 2 saal 12.5% -> 13.3%; Gali 12.7% -> 13.7%, Disawar 11.4% -> 12.6%), "
             "Top-25 28.6% -> 29.7% (Gali 30.1% -> 31.6%)"},
    {"version": "5.0 (test kiye, nahi jode)", "date": "2026-10-05",
     "old": "-",
     "failure": "-",
     "discovery": "Momentum (badhti ginti, t = +5.0), overdue (apne ausat gap se zyada door, t = -8.3), "
                  "complement / mirror (t = +1.4 / -2.7), Andar/Bahar digit transitions",
     "change": "Nahi jode: momentum aur overdue gap/balancing ko hi doosre naam se naapte hain (self-critic: "
               "duplicate pattern)",
     "test": "Top-10 momentum ke saath 11.71%, overdue 11.80%, complement 11.98%, chaaron saath 12.07% "
             "(sab 12.49% se kam); Andar/Bahar transitions ke saath 51.5 / 52.8% (bina 51.8 / 53.1%)"},
]


def rank_calibration(pe) -> dict:
    """How often the real number had each rank in the walk-forward test (per rank bucket)."""
    st = pe.st
    idx = np.arange(START, len(st))
    order = np.argsort(-pe.jodi.P[idx], axis=1, kind="stable")
    ranks = np.argmax(order == st.vals[idx][:, None], axis=1) + 1
    n = len(ranks)
    out = []
    for lo, hi in BUCKETS:
        k = int(((ranks >= lo) & (ranks <= hi)).sum())
        size = hi - lo + 1
        per = k / max(n, 1) / size
        out.append({"from": lo, "to": hi, "hits": k, "n": n, "per_number": round(per, 5),
                    "p_value": round(float(binom_sf(k, n, size / 100)), 6)})
    return {"draws": n, "buckets": out}


def _bucket(calib: dict, rank: int) -> dict:
    return next(b for b in calib["buckets"] if b["from"] <= rank <= b["to"])


def _explain(pe, market: str, date: dt.date):
    """Pattern values, weights and each number's per-pattern push for the next draw."""
    st = pe.st
    i = st.index_of(market, date)
    tr = (pe._tr_all if i >= len(st) else Transitions.upto(st, i)) if pe.transitions else None
    z = _zs(pe._jodi_raw(i, market, date, tr))
    cols, w = pe.jodi.refit(min(i, len(pe.jodi.y)))
    p = probs(z, cols, w)
    rel = z[:, cols] - p @ z[:, cols]                  # number's pattern value vs the average number
    return i, [pe.jodi.names[c] for c in cols], rel * w, rel


def _signals(names, push_row, rel_row, k=2, eps=0.005):
    """Patterns that pushed a number up / down, each with which way its value was."""
    def said(n, r):
        return f"{LABELS.get(n, n)} ({'zyada' if r > 0 else 'kam'})"
    up = sorted(((n, x, r) for n, x, r in zip(names, push_row, rel_row) if x > eps), key=lambda t: -t[1])
    down = sorted(((n, x, r) for n, x, r in zip(names, push_row, rel_row) if x < -eps), key=lambda t: t[1])
    return [said(n, r) for n, _, r in up[:k]], [said(n, r) for n, _, r in down[:k]], len(up)


def engine5_report(pe, market: str, date: dt.date, locked: dict, calib: dict) -> dict:
    st = pe.st
    dist = np.array(locked["dist"])
    top = [v for v, _ in locked["top10"]]
    order = top + [int(v) for v in np.argsort(-dist, kind="stable") if int(v) not in top]
    rank = {v: r + 1 for r, v in enumerate(order)}
    lp = np.log(np.maximum(dist, 1e-12))
    score = 100 * (lp - lp.min()) / max(lp.max() - lp.min(), 1e-12)
    i, names, push, rel = _explain(pe, market, date)

    def sig(v):
        return _signals(names, push[v], rel[v])

    def cand(v):
        up, down, n_up = sig(v)
        b = _bucket(calib, rank[v])
        return {"number": int(v), "rank": rank[v], "score": int(round(score[v])),
                "prob": round(b["per_number"] * 100, 2),
                "confidence": "Medium" if b["p_value"] < 0.01 and b["per_number"] > 0.011 else "Low",
                "evidence": up, "against": down, "support": n_up}

    top_c = [cand(v) for v in order[:10]]
    vals = st.vals[:i]
    # gap: results since each number last came, and its own average gap so far
    last = np.full(100, -1)
    gsum, gcnt = np.zeros(100), np.zeros(100)
    for j, v in enumerate(vals):
        if last[v] >= 0:
            gsum[v] += j - last[v]
            gcnt[v] += 1
        last[v] = j
    gap = np.where(last >= 0, i - last, i + 1)
    avg = np.where(gcnt > 0, gsum / np.maximum(gcnt, 1), 100.0)
    ratio = gap / avg
    rising = np.bincount(vals[-100:], minlength=100) / 100 - np.bincount(vals[-500:-100], minlength=100) / 400
    hot = np.bincount(vals[-60:], minlength=100)
    recent = [int(v) for v in vals[-4:][::-1]]
    rev = []
    for v in recent:
        r_ = (v % 10) * 10 + v // 10
        if r_ != v and r_ not in [x["number"] for x in rev]:
            rev.append({"number": r_, "from": v, "rank": rank[r_]})
    tr_col = names.index("tr_market") if "tr_market" in names else None
    hidden = sorted((v for v in order[10:30]), key=lambda v: (-sig(v)[2], rank[v]))[:5]
    contrarian = (sorted(order[10:40], key=lambda v: -push[v, tr_col])[:5] if tr_col is not None else [])
    rejected = [v for v in np.argsort(-(hot + np.arange(100) * 1e-6)) if hot[v] >= 2 and rank[int(v)] > 10][:5]
    best = top_c[0]
    b1 = _bucket(calib, 1)
    t10 = calib["buckets"][0]["hits"] + calib["buckets"][1]["hits"]
    return {
        "market": market, "date": date.isoformat(),
        "top": top_c,
        "strongest": {**best, "why": (f"{best['number']:02d} ko sabse zyada patterns ne upar rakha: "
                                      + ("; ".join(best["evidence"]) or "koi ek bada pattern nahi")
                                      + (f". Virodh: {'; '.join(best['against'])}." if best["against"] else "."))},
        "hidden": [{**cand(v), "note": "kai patterns ka saath, par " + (", ".join(sig(v)[1]) or "baaki") + " ne neeche rakha"} for v in hidden],
        "gap": [{"number": int(v), "gap": int(gap[v]), "avg_gap": round(float(avg[v]), 1), "rank": rank[int(v)]}
                for v in np.argsort(-ratio, kind="stable")[:5]],
        "momentum": [{"number": int(v), "last100": int(np.bincount(vals[-100:], minlength=100)[v]),
                      "rate_change": round(float(rising[v]) * 100, 2), "rank": rank[int(v)]}
                     for v in np.argsort(-rising, kind="stable")[:5]],
        "reverse": rev[:4],
        "contrarian": [{**cand(v), "note": "transition pattern iske paksh me (is market me pichle number ke baad yeh jump abhi tak nahi hua)"}
                       for v in contrarian],
        "rejected": [{"number": int(v), "rank": rank[int(v)], "recent60": int(hot[v]),
                      "why": ", ".join(sig(int(v))[1]) or "patterns ka saath nahi"} for v in rejected],
        "uncertainty": (f"Sabse upar wale number ka bhi asli chance sirf ~{b1['per_number'] * 100:.2f}% hai (random 1%); "
                        f"Top-10 me asli number 5 saal me {t10 / calib['draws'] * 100:.1f}% baar aaya, yaani ~{100 - t10 / calib['draws'] * 100:.0f}% din MISS. "
                        "Rank 1 aur rank 10 ka fark bahut chhota hai."),
        "changed": ["Naya: transition patterns (isi market me pichle number ke baad ka jump, koi bhi market ka pichla "
                    "number, digit -> digit) — 5 saal me sabit, pehle kisi engine me nahi tha.",
                    "Test karke chhode: momentum, overdue gap, complement/mirror (inse Top-10 ghatti thi).",
                    "Andar/Bahar: Pattern Engine wale digit models (transitions se behtar nahi hue).",
                    "Report: har number ka score 0-100, maapi hui probability, saath/virodh wale patterns."],
    }
