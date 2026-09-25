# Satta Predictor — self-learning prediction engine

Disawar, **Faridabad**, Ghaziabad aur Gali ke results par chalne wala tool. Yeh
khud data laata hai, khud agla number predict karta hai, result aane se pehle
prediction ko **lock** karta hai (time + SHA-256 hash), result aane ke baad khud
milata hai ki sahi tha ya galat, galti ka reason likhta hai aur apne models ko
khud sudharta hai. Aapko kuch feed nahi karna.

> **Seedhi baat:** agar numbers sach me random hain to koi bhi formula unhe
> pakka predict nahi kar sakta. Isliye tool har din apni asli accuracy random
> chance se compare karta hai (exact = 1%, top-10 = 10%, Andar/Bahar top-3 = 30%)
> aur p-value dikhata hai. Agar data me koi asli pattern hoga to tool use khud
> pakad lega (tests me yeh prove kiya gaya hai). Agar nahi hoga to tool khud
> bata dega ki woh random se behtar nahi hai.

---

## 1. Chalane ka tareeka (apne computer par)

Python 3.10+ chahiye.

```bash
git clone <this repo> && cd predict-satta
pip install -r requirements.txt
python -m satta serve
```

Browser me kholo: **http://localhost:8000**

Pehli baar server khud Jan 2026 se aaj tak ka data websites se download karega
(1-2 minute). Uske baad har 15 minute me naya result check karta hai. "Abhi
Update Karo" button dabane se turant cycle chalta hai.

### Commands

| Command | Kya karta hai |
|---|---|
| `python -m satta serve` | Dashboard + backend + automatic scheduler |
| `python -m satta cycle` | Ek baar: fetch → check → seekho → predict → lock |
| `python -m satta predict` | Har market ki locked agli prediction |
| `python -m satta backtest --market faridabad --days 7` | 7 din ka test, har din ka hit/miss + reason |
| `python -m satta table` | Jan 2026 se aaj tak ka poora result table |
| `python -m satta formulas` | Tool ke khud ke formule + unseen data par test |
| `python -m satta theorems` | Statistical findings (pattern hai ya random) |
| `python -m satta verify` | Har locked prediction ka hash proof check |
| `python -m satta sync --full` | Saara data dobara download |
| `python -m satta import file.csv` | Backup: agar websites band hon to CSV se data (`date,market,value`) |

## 2. Automation (bina computer on rakhe)

`.github/workflows/auto-predict.yml` GitHub Actions par **har ghante** chalta hai
(repo ki default branch par merge hone ke baad):

1. naya result download
2. purani locked prediction ko result se milana
3. models ke weights update (self-correction)
4. agle result ki prediction lock karke `data/predictions.jsonl` me commit

Har commit ka time GitHub par public record hai — yeh proof hai ki prediction
result se pehle bani. Actions tab me "Auto predict" → "Run workflow" se turant bhi
chala sakte ho.

Dashboard online dekhne ke liye: repo **Settings → Pages → Deploy from branch →
main / (root)**. Phir `https://<username>.github.io/predict-satta/` khulega
(static mode, har ghante update).

## 3. Dashboard

| Tab | Kya dikhta hai |
|---|---|
| **Aaj ki Prediction** | Har market ki agli Top-10 jodi (probability ke saath), Andar/Bahar top-3, tool ke formule, countdown, lock time, SHA-256 hash |
| **Proof (Live)** | Result se pehle locked predictions vs asli result: HIT/MISS, rank, "kyu galat hua / kya seekha", hash verified |
| **7-Din Test** | Walk-forward backtest: pichle 7 / 30 / saare din, har din ka reason |
| **Learning** | 20 models ke weights ka graph — kaunsa model kab sahi nikla aur uska bharosa kaise badla |
| **Formule** | Tool ke dhoondhe formule, train vs unseen test, p-value |
| **Theorems** | Data par statistical tests: uniformity, independence, serial correlation, runs, cross-market, entropy, Hedge theorem |
| **Result Chart** | Jan 2026 se aaj tak har din ka har market ka result, CSV download |

## 4. Math — engine kaise sochta hai

### 4.1 Twenty experts (har ek alag theory)

Har expert `P(agla number = v)` deta hai, v = 00..99.

| Expert | Formula |
|---|---|
| Random baseline | `P(v) = 1/100` (control) |
| Frequency (Bayes) | `P(v) = (n_v + a) / (n + 100a)` — Dirichlet posterior, a = 1, 20 |
| Recency | `P(v) ∝ Σ_t 2^(−age_t / h)·[y_t = v]`, h = 7, 30, 90 din |
| Hot Andar-Bahar | `P(v) = P(andar)·P(bahar)`, decayed digit counts, h = 5, 20 |
| Markov jodi | `P(v | x) = (C[x→v] + k·P_digit(v)) / (C[x] + k)` back-off ke saath |
| Markov digit | `P(andar_t | andar_{t−1})·P(bahar_t | bahar_{t−1})` (+ palti-cross variant) |
| Gap hazard | survival analysis: `h(g) = events_g / exposure_g`, `P(v) ∝ h(gap_v)` |
| Cross-market | `P(andar | andar of M kal)·P(bahar | bahar of M kal)`, M = doosre markets |
| Weekday | hafte ke din ke hisaab se digit distribution |
| Satta tricks | palti, cut (+5), ±1, ±10, ±11, 99−x, jod — har rule ka Beta-posterior hit-rate |
| Pattern match | method of analogues: pichle 2/3 din jaise itihaas ke baad kya aaya |
| Spectral | periodogram se top-3 cycles, least-squares harmonic fit, 1 din aage extrapolate |
| Formula (jodi / digit) | tool ke khud ke formule (neeche), har 7 din par dobara search |

### 4.2 Self-correction (Fixed-Share Hedge)

```
P_t(v)      = Σ_i w_i · P_i(v)                    final prediction
w_i        ← w_i · P_i(asli number)               result aane ke baad
w_i        ← (1 − α)·w_i/Σw + α/N                 α = 1%, taaki koi model hamesha ke liye mar na jaye
```

**Theorem (Bayes mixture):** `L_mix ≤ min_i L_i + ln N` — cumulative log-loss me
ensemble kabhi best single model se `ln 20 ≈ 3` se zyada peeche nahi rehta.
Dashboard ka T10 is theorem ko asli data par check karta hai.

### 4.3 Khud ke formule

Tool ~1000 formule banata hai:

```
jodi  = (c1·U + c2·V + k) mod 100
andar = (c1·u + c2·v + k) mod 10
bahar = (c1·u + c2·v + k) mod 10
```

U, V = pichla number, 2 din pehle ka, doosre markets ka kal ka number, unka ulta,
tareekh; u, v = inke digits, hafte ka din, mahina; c ∈ {+1, −1}; k = sabse zyada
baar aane wala residual. Pehle 70% din par formula dhoondha jaata hai, aakhri
30% (unseen) par test hota hai aur exact binomial test se chance se compare
hota hai. Sirf wahi "asli pattern" kehlata hai jo unseen data par bhi chale.

### 4.4 Walk-forward (no cheating)

Backtest me din `t` ki prediction sirf `t` se pehle ke data se banti hai —
bilkul live jaisa. `tests/test_engine.py::test_no_future_leak` isko check karta hai.

## 5. Proof system

- Har prediction `data/predictions.jsonl` me append hoti hai, kabhi badli nahi jaati.
- `hash = SHA-256(market, date, created_at, top10, dist)`; badlav hua to dashboard "TAMPERED" dikhayega.
- `late: true` = result time ke baad bani, isko accuracy me nahi gina jaata.

## 6. Data sources

`satta/config.py` → `DEFAULT_SOURCES` (satta-king-fast.com monthly chart + teen
Faridabad yearly charts). Parser kisi bhi table layout ko samajhta hai (market
columns, month columns, date list). Kai sources majority vote se merge hote hain.
Naya source add karna ho to `data/sources.json` me same format me likho.

## 7. Technology

| Layer | Tech |
|---|---|
| Engine | Python 3, NumPy (Bayesian stats, Markov chains, survival analysis, spectral analysis, online learning) |
| Backend | FastAPI + Uvicorn, background scheduler thread |
| Scraper | requests + BeautifulSoup (layout-agnostic table parser) |
| Storage | plain CSV/JSONL (git-friendly proof) |
| Frontend | HTML + CSS + vanilla JS, SVG charts (koi build step nahi, offline chalta hai) |
| Automation | GitHub Actions (hourly), GitHub Pages (optional) |
| Tests | pytest — parser, no-leak, planted-pattern learning, locking, tamper detection |

```
satta/
  config.py        markets, result times, sources
  scraper.py       fetch + parse + merge
  storage.py       results.csv, predictions.jsonl, hashes
  service.py       daily cycle, dashboard payload
  api.py           FastAPI backend
  engine/
    base.py        series, features, context (sirf past data)
    experts.py     20 experts
    formulas.py    formula discovery + holdout test
    ensemble.py    Fixed-Share Hedge, replay, scoring, post-mortem
    theorems.py    statistical findings
    stats.py       binomial / chi-square / normal
web/               dashboard
tests/             pytest
```

Test: `python -m pytest -q`
