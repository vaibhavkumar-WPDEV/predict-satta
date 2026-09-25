"""The engine's own findings ("theorems"), each backed by a statistical test
on the market's real history. A pattern is only called a pattern when its
p-value passes a Bonferroni-corrected threshold, so the engine does not fool
itself with coincidences.
"""

from __future__ import annotations

import math

import numpy as np

from .base import SeriesData, rev
from .stats import binom_sf, chi2_sf, norm_sf

ALPHA = 0.05


def _chi2_uniform(counts: np.ndarray) -> tuple[float, int]:
    n = counts.sum()
    e = n / len(counts)
    return float(((counts - e) ** 2 / e).sum()), len(counts) - 1


def _chi2_indep(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    M = np.zeros((10, 10))
    np.add.at(M, (a, b), 1)
    M = M[M.sum(1) > 0][:, M.sum(0) > 0]
    E = np.outer(M.sum(1), M.sum(0)) / M.sum()
    return float(((M - E) ** 2 / E).sum()), (M.shape[0] - 1) * (M.shape[1] - 1)


def findings(sd: SeriesData, replay=None, n_experts: int | None = None) -> list[dict]:
    y = sd.y
    n = len(y)
    if n < 30:
        return []
    t, u = y // 10, y % 10
    out: list[dict] = []

    def add(fid, title, statement, stat, p, note=""):
        out.append({"id": fid, "title": title, "statement": statement, "stat": stat,
                    "p_value": p, "note": note})

    chi, dof = _chi2_uniform(np.bincount(y, minlength=100))
    add("T1", "Jodi uniformity (χ²)",
        "H0: har jodi 00-99 ki probability barabar (1%) hai.",
        f"χ²={chi:.1f}, dof={dof}, n={n}", chi2_sf(chi, dof),
        "" if n >= 500 else "n<500: har jodi ka expected count 5 se kam, test kamzor hai.")
    for name, d in (("Andar", t), ("Bahar", u)):
        chi, dof = _chi2_uniform(np.bincount(d, minlength=10))
        add("T2" + name[0], f"{name} digit uniformity (χ²)",
            f"H0: {name} digit 0-9 sab barabar (10%) aate hain.",
            f"χ²={chi:.1f}, dof={dof}", chi2_sf(chi, dof))
    chi, dof = _chi2_indep(t, u)
    add("T3", "Andar–Bahar independence (χ²)",
        "H0: Andar digit aur Bahar digit ek doosre se independent hain.",
        f"χ²={chi:.1f}, dof={dof}", chi2_sf(chi, dof))

    r = float(np.corrcoef(y[:-1], y[1:])[0, 1]) if np.std(y) > 0 else 0.0
    z = r * math.sqrt(n - 1)
    add("T4", "Serial correlation (lag-1)",
        "H0: aaj ka number kal ke number se linearly related nahi hai (ρ=0).",
        f"r={r:+.3f}, z={z:+.2f}", 2 * norm_sf(abs(z)))

    above = y > np.median(y)
    runs = 1 + int((above[1:] != above[:-1]).sum())
    n1, n2 = int(above.sum()), int((~above).sum())
    if n1 and n2:
        mu = 2 * n1 * n2 / (n1 + n2) + 1
        var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
        zr = (runs - mu) / math.sqrt(var) if var > 0 else 0.0
        add("T5", "Runs test (Wald–Wolfowitz)",
            "H0: median se upar/neeche ka sequence random hai (na streaks, na zig-zag).",
            f"runs={runs}, expected={mu:.1f}, z={zr:+.2f}", 2 * norm_sf(abs(zr)))

    rep_hits = int((y[1:] == y[:-1]).sum())
    add("T6", "Repeat (same jodi next draw)",
        "H0: agle draw me wahi jodi aane ka chance 1% hai.",
        f"{rep_hits}/{n - 1} baar", binom_sf(rep_hits, n - 1, 0.01))
    pal_hits = int((y[1:] == rev(y[:-1])).sum())
    add("T7", "Palti (ulta jodi next draw)",
        "H0: agle draw me palti jodi aane ka chance 1% hai.",
        f"{pal_hits}/{n - 1} baar", binom_sf(pal_hits, n - 1, 0.01))

    for m in sd.others:
        x = sd.F[m]
        ok = x >= 0
        if ok.sum() < 30:
            continue
        chi, dof = _chi2_indep(x[ok] // 10, t[ok])
        add("T8" + m[:2].upper(), f"Cross-market: {m.title()} (kal) → andar",
            f"H0: {m.title()} ka kal ka andar digit, aaj ke andar digit ko affect nahi karta.",
            f"χ²={chi:.1f}, dof={dof}, n={int(ok.sum())}", chi2_sf(chi, dof))

    counts = np.bincount(y, minlength=100)
    p = counts[counts > 0] / n
    h = float(-(p * np.log2(p)).sum() + (len(p) - 1) / (2 * n * math.log(2)))
    add("T9", "Entropy (Miller–Madow)",
        "Maximum possible randomness log2(100) = 6.644 bits. Jitna paas, utna unpredictable.",
        f"H ≈ {h:.3f} bits ({h / math.log2(100):.1%} of max)", None)

    m_tests = sum(1 for f in out if f["p_value"] is not None)
    thr = ALPHA / max(m_tests, 1)
    for f in out:
        if f["p_value"] is None:
            f["verdict"] = "info"
            f["pattern"] = None
        elif f["p_value"] < thr:
            f["verdict"] = f"PATTERN MILA (p < {thr:.4f}, Bonferroni)"
            f["pattern"] = True
        else:
            f["verdict"] = "Random jaisa — pattern sabit nahi hua"
            f["pattern"] = False

    if replay is not None and replay.steps:
        best = float(replay.losses.min())
        k = n_experts or len(replay.experts)
        bound = best + math.log(k)
        steps = len(replay.steps)
        out.append({
            "id": "T10", "title": "Hedge regret theorem (check on real data)",
            "statement": "Theorem: L_mix ≤ min_i L_i + ln N  (Bayes mixture, log-loss). "
                         "Ensemble kabhi best model se ln N se zyada peeche nahi hoga.",
            "stat": f"L_mix={replay.mix_loss:.1f}, best={best:.1f}, bound={bound:.1f}, "
                    f"random={steps * math.log(100):.1f} ({steps} din)",
            "p_value": None, "pattern": None,
            "verdict": "Theorem data par sahi" if replay.mix_loss <= bound + 1e-6 else
                       "Fixed-share alpha ki wajah se thoda upar (expected)",
            "note": "Agar L_mix ≈ random, to koi bhi model chance se behtar nahi hai.",
        })
    return out
