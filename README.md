# Satta Predictor — self-learning prediction engine

Disawar, **Faridabad**, Ghaziabad aur Gali ke results par chalne wala tool. Yeh
khud data laata hai, khud agla number predict karta hai, result aane se pehle
prediction ko **lock** karta hai (time + SHA-256 hash), result aane ke baad khud
milata hai ki sahi tha ya galat, galti ka reason likhta hai aur apne models ko
khud sudharta hai. Aapko kuch feed nahi karna.

> **Seedhi baat:** agar numbers sach me random hain to koi bhi formula unhe
> pakka predict nahi kar sakta. Isliye tool har din apni asli accuracy random
> chance se compare karta hai (exact = 1%, top-10 = 10%, Andar/Bahar top-5 = 50%)
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

Pehli baar server khud Jan 2021 se aaj tak (5 saal) ka data websites se download karega
(1-2 minute). Uske baad har 15 minute me naya result check karta hai. "Abhi
Update Karo" button dabane se turant cycle chalta hai.

### Commands

| Command | Kya karta hai |
|---|---|
| `python -m satta serve` | Dashboard + backend + automatic scheduler |
| `python -m satta cycle` | Ek baar: fetch → check → seekho → predict → lock |
| `python -m satta predict` | Har market ki locked agli prediction |
| `python -m satta backtest --market faridabad --days 7` | 7 din ka test, har din ka hit/miss + reason |
| `python -m satta table` | Jan 2021 se aaj tak ka poora result table |
| `python -m satta formulas` | Tool ke khud ke formule + unseen data par test |
| `python -m satta theorems` | Statistical findings (pattern hai ya random) |
| `python -m satta verify` | Har locked prediction ka hash proof check |
| `python -m satta sync --full` | Saara data dobara download |
| `python -m satta import file.csv` | Backup: agar websites band hon to CSV se data (`date,market,value`) |

## 2. Automation (bina computer on rakhe)

`.github/workflows/auto-predict.yml` GitHub Actions par **har 30 minute** chalta hai
(repo ki default branch par merge hone ke baad):

1. naya result download
2. purani locked prediction ko result se milana
3. models ke weights update (self-correction)
4. agle result ki prediction lock karke `data/predictions.jsonl` me commit

Har commit ka time GitHub par public record hai — yeh proof hai ki prediction
result se pehle bani. Actions tab me "Auto predict" → "Run workflow" se turant bhi
chala sakte ho.

Dashboard online dekhne ke liye: repo **Settings → Pages → Deploy from branch →
`claude/satta-prediction-tool-0q71fv` (ya jo default branch ho) / (root)**. Phir
`https://<username>.github.io/predict-satta/` khulega (static mode, har ghante
update). Private repo par Pages ke liye GitHub ka paid plan chahiye; warna
`python -m satta serve` se local chalao.

## 3. Dashboard

| Tab | Kya dikhta hai |
|---|---|
| **Aaj ki Prediction** | Har market ki agli Top-10 jodi (probability ke saath), Andar/Bahar top-5, tool ke formule, countdown, lock time, SHA-256 hash |
| **Proof (Live)** | Result se pehle locked predictions vs asli result: HIT/MISS, rank, "kyu galat hua / kya seekha", hash verified |
| **7-Din Test** | Walk-forward backtest: pichle 7 / 30 / saare din, har din ka reason |
| **Learning** | 20 models ke weights ka graph — kaunsa model kab sahi nikla aur uska bharosa kaise badla |
| **Formule** | Tool ke dhoondhe formule, train vs unseen test, p-value |
| **Theorems** | Data par statistical tests: uniformity, independence, serial correlation, runs, cross-market, entropy, Hedge theorem |
| **History (sab predictions)** | Chaaron markets ki har locked prediction: Top-10, Andar top-5, Bahar top-5, asli result, HIT/MISS, market-wise score; `data/prediction_history.csv` me bhi har cycle par likhi jaati hai |
| **Result Chart** | 2021 se aaj tak har din ka har market ka result; jis din prediction thi wahan J/A/B ✔✘, CSV download |

## 4. Math — engine kaise sochta hai

### 4.1 28 experts (har ek alag theory)

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
| Spectral (Fourier) | periodogram se top-3 cycles, least-squares harmonic fit, 1 din aage extrapolate |
| Formula (jodi / digit) | tool ke khud ke formule (neeche), har 7 din par dobara search |
| Aryabhata kuttaka | sab 10,000 linear congruences `y_t ≡ a·y_(t−1) + c (mod 100)`; digits par 2nd order `d_t ≡ a·d_(t−1) + b·d_(t−2) + c (mod 10)`. Kisi LCG random generator ko crack kar deta hai (test me 90%+ exact) |
| Fibonacci / Golden ratio (Pingala, Da Vinci) | `y_t = (y_(t−1)+y_(t−2)) mod 100`, golden rotation `+100/φ`, Fibonacci lag echo `y_t = y_(t−F)` |
| Vedic beejank / Tesla 3-6-9 | digital root `dr(x) = 1 + (x−1) mod 9` par Markov chain |
| Einstein Brownian motion | random walk: badlaav `Δ = y_t − y_(t−1)` ka smoothed distribution |
| Universal prediction (CTW) | Context Tree Weighting (Willems 1995): andar/bahar ke har context 0–3 digits ka KT estimator, saare Markov orders ka exact Bayesian average |
| Neural network | 1 hidden layer (32 tanh), input = pichle draws + doosre markets ke digits + weekday, online SGD + experience replay |
| Genetic programming | tool ke khud ke formule: `+ − ×`, ulta, cut, andar/bahar, jod, beejank, jodi(a,b) se bane expression trees; selection + crossover + mutation se evolve, pichle 730 draws par |

Formula-experts ka bharosa adaptive hai: `P = (1−g)·uniform + g·votes`, jahan `g` = best
formula ka purana hit-rate (5–95%). Random data par g ≈ 5% rehta hai, asli formula par 90%+.

### 4.2a Meta-learning (self-tuning)

Learning speed `η ∈ {0.5, 1, 2}` aur bhoolne ki dar `α ∈ {0.002, 0.01, 0.05}` ki 9 settings
saath chalti hain; upar ek aur Hedge unhe weight deta hai. Final prediction phir bhi ek hi
mixture hai: `w_eff = Σ_g v_g · w_g`. Dashboard Learning tab me dikhta hai tool ne kaunsi
setting chuni.

### 4.2a2 Calculations ka merge

Experts ki raay do tareeke se merge hoti hai: linear pool `Σ w_i P_i` aur geometric pool
`∝ Π P_i^(w_i)` (jahan sab sahmat hon wahan tez). Ek aur Hedge results dekh kar tay karta hai
kitna kaunsa. Heavy formula-models pichle 730 draws (≈2 saal) par chalte hain, frequency /
Markov / CTW / neural network poore 5 saal par.

### 4.2a3 Markets ka same-day connection

Din me results is kram me aate hain: Disawar ~05:00 → Faridabad ~18:15 → Ghaziabad ~21:30 →
Gali ~23:30. Kisi market ki prediction apne pichle result ke baad aaye doosre markets ke
har result ka intezaar karti hai — usi din ke pehle wale (jaise Faridabad subah ke Disawar ka)
aur pichli raat ke baad wale (jaise Disawar raat ke Faridabad, Ghaziabad, Gali ka; engine 4.0
se) — taaki sabse taaza number bhi calculation me jude, bilkul walk-forward test jaisa. Der ho
to result time se 20 minute pehle bina uske lock ho jaati hai (Ghaziabad ~22:23 tak aata hai, isliye Gali 23:10 tak rukti hai). Theorems tab me "(aaj)" wale tests yeh connection check
karte hain.

### 4.2a4 Miss-correction

Har result ke baad tool dekhta hai ki asli number uski list se kis tarah khiska tha (palti,
±1, ±10, ±11, cut). Ek Hedge in shifts ko weight deta hai; agar koi khiskav baar-baar sahi
nikle to agli poori list usi taraf khisak jaati hai (`P'(v) = Σ_k c_k · P(T_k⁻¹(v))`).
Test me: jo model hamesha 1 se chookta tha, uske saath tool ne "+1" seekh kar 100% exact kiya.

### 4.2a5 Hot pool + Top-10 selector (pehla asli fayda)

- **Hot pool (number ghoom ke aata hai):** 2021–26 data me jo number pichle ~7 din me kisi bhi
  market me aaya, woh agle result me random se thoda zyada aata hai (Disawar 28.0% vs 23.4%,
  z = +4.8; Ghaziabad z = +4.4; Gali z = +3.2; Faridabad me asar nahi). Yeh model chaaron markets
  ke haal ke numbers ko `0.85^age` se score karta hai; kitna bharosa (λ) karna hai woh roz
  pichle 730 draws se khud seekhta hai.
- **Top-10 selector:** ensemble apne models ko log-loss se tolta hai, lekin lakshya "asli number
  Top-10 me" hai. Har result ke baad har model (aur ensemble) ki Top-10 check hoti hai,
  `score += 0.02·(hit − 0.1)`, aur agli list sabse achhe score wale ki hoti hai (walk-forward).
- **5 saal ka walk-forward nateeja (engine 3.2):** Disawar 11.4% (p = 0.02), Gali 10.7%,
  Ghaziabad 10.0%, Faridabad 9.5% — random 10%. Engine 3.1 par sab 8.7–9.3% the.

### 4.2a6 Decision engine, calibration, symbolic layer (engine 3.3)

- **Signal quality** har model ka: Strong (p<0.01, dono aadhon me >10.5%), Moderate (p<0.05,
  dono >10%), Weak (10% se upar par sabit nahi), Unreliable.
- **EDGE / NO EDGE:** market ki final Top-10 ka 5 saal ka walk-forward record Strong/Moderate ho
  tabhi "EDGE", warna "Insufficient predictive edge". Abhi: Disawar EDGE (Moderate, 11.4%,
  dono aadhe 11.4%), Faridabad / Ghaziabad / Gali NO EDGE.
- **Calibrated probability:** list me asli number aane ka chance = pichle 730 din ka tested
  hit-rate + 90% Wilson range (model ke apne dawe ki jagah).
- **Monte Carlo (binomial):** agle 30 din me kitne hit ki ummeed, random se tulna.
- **Numerology + Chandra tithi:** 14 rules (mulank, bhagyank, DD+MM, tithi …) — asli data par
  0.83–0.94% sahi (random 1%), tithi ka andar par asar nahi (T11/T12). Weight data tay karta hai.
- **Cyclical time** (sin/cos of tareekh, mahina, din) neural network me.
- **Arrival window:** watcher ke dekhe samay se "result aam taur par kab aata hai".
- Mausam ka data uplabdh nahi; tarot random hai — dono use nahi hote.

### 4.2a7 Engine 4.0: galtiyon ki jaanch ke baad (29 Sep 2026)

Live misses ki jaanch me do cheezein mili:

1. **Galat data:** 28 Sep 21:34 par website Ghaziabad = 48 dikha rahi thi, 22:18 par 03 kiya.
   Gali 28 ki prediction us galat 48 se lock hui. Ab pichle 2 din ka koi bhi naya result, jo sirf
   ek website par ho, tabhi use hota hai jab agle fetch me (10 min baad) wahi number dobara dikhe;
   2 websites mile to turant. (`scraper.merge`, `data/sync_state.json` → `pending`)
2. **Asli pattern dheela ho raha tha:** 5 saal me sirf ek pattern sabit hua — jo number haal me
   kisi bhi market me aaya, woh thoda zyada dobara aata hai (pichle 3 results 1.22x). 30 models
   ka merge ise dheela kar deta tha. Engine 4.0 me Top-10 seedhe is recency se banti hai
   (`satta/engine/recency.py`): score = Σ λ^age · ([number aaya] + palti_w · [palti aaya])
   pichle 60 results (chaaron markets) par. 12 settings (λ ∈ 0.5…0.95, palti 0/0.3) saath chalti
   hain; har result ke baad har setting ka HIT/MISS record hota hai aur agli list us setting se
   banti hai jiske pichle 1460 results me sabse zyada HIT — future data se kuch tune nahi hota.

Same 5 saal ke data par walk-forward Top-10: Disawar 12.2%, Faridabad 10.6%, Ghaziabad 12.1%,
Gali 11.5%, chaaron 11.57% (p = 0.000003); engine 3.3 par 10.54% tha. Andar/Bahar top-5 ~49–53%.
Har MISS ke "kyu?" me ab likha hota hai: number pichle 60 results me aaya tha ya nahi, kitni
settings ne use pakda, aur agli list ki setting kya hui.

### 4.3 Pattern Engine — upgraded level (alag switch, engine 4.0 par koi asar nahi)

Website par upar **Engine** switch: "Engine 4.0 (abhi wala)" ya "Pattern Engine (upgraded level)"
(link: `web/?engine=pattern`). Pattern Engine ka code `satta/pattern/`, files `data/pattern/`
(apni hash-locked predictions, history CSV, dashboard). Watcher me alag step
(`python -m satta pattern`) — fail ho to bhi engine 4.0 chalta rahe.

- Chaaron markets ek line me (declaration order). Har number ke liye 19 jodi patterns: recency
  (3 speed), palti, ±1, cut, andar/bahar digit, isi market ka pichla number / palti, **time patterns**
  (isi weekday 8 hafte, isi tareekh 3 mahine, weekday ke andar/bahar digit 2 saal), 1 saal ki
  market ginti, 1460 result ki ginti (balancing), gap, aaj pehle aaye market.
- **Pattern gate:** har refit par har pattern ka score test pichle 2920 results par
  (t = asli number par pattern ka standardised maan / standard error). |t| > 2 ho tabhi model me.
- **Model:** conditional logit P(v) ∝ exp(Σ w·z) — Newton se maximum likelihood (ridge L2),
  walk-forward har 200 draws, live me har naye result ke baad dobara fit. Andar/Bahar ke liye
  10-pattern digit models (sab patterns, weights data tay karta hai).
- 5 saal ka sabse bada asar: **balancing** (jo number 4 saal me zyada aaye, woh kam aate hain,
  t = −7.6; sabse zyada aaye 10 numbers sirf 8.1% aate hain), **gap** (t = −6.8), recency
  (+4.5), digit recency (+3.6), palti (+3.2). Time patterns (weekday, tareekh) |t| < 1.4 —
  abhi tak koi asar nahi; gate inhe har refit par dobara test karta hai.

Same 7,739 draws (2021–26, walk-forward), engine 4.0 → Pattern Engine: jodi Top-10 11.5% →
**11.95%** (p ≈ 1e-8; pichle 2 saal 12.5%; Ghaziabad 13.0%, Gali 12.7%, Faridabad 10.6%,
Disawar 11.4% jahan 4.0 ka 12.2% behtar hai), Andar top-5 50.7% → **51.8%**, Bahar top-5
51.3% → **53.1%**.

### 4.2b Khud ko todo + seekhne ka asar

- **Shuffle test:** engine ko 3 baar aisi history par chalaya jaata hai jisme dates ka order
  shuffle hai (har number utni hi baar, par koi time-pattern nahi). Asli data par score
  shuffled se behtar nahi → engine ne koi asli pattern nahi pakda.
- **Learned vs equal weights:** wahi models bina seekhe (barabar weight) vs Hedge se seekh kar —
  dashboard ke Learning tab me rolling 50-din top-10 rate ka graph.

### 4.2 Self-correction (Fixed-Share Hedge)

```
P_t(v)      = Σ_i w_i · P_i(v)                    final prediction
w_i        ← w_i · P_i(asli number)               result aane ke baad
w_i        ← (1 − α)·w_i/Σw + α/N                 α = 1%, taaki koi model hamesha ke liye mar na jaye
```

**Theorem (Bayes mixture):** `L_mix ≤ min_i L_i + ln N` — cumulative log-loss me
ensemble kabhi best single model se `ln 28 ≈ 3.3` se zyada peeche nahi rehta.
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

Data checks jo har fetch par chalte hain:

- har website ka raw data `data/raw/<source>.csv` me alag save hota hai
- sources ko aapas me same din aur ±1 din shift par milaya jaata hai (dashboard `sync.agreement`)
- tareekh jaisa dikhne wala column aur result-time se pehle ka "result" reject hota hai
- chhutti ke din data se seekhe jaate hain (abhi: Faridabad/Ghaziabad/Gali — mahine
  ka aakhri din, Disawar — 1 tareekh) aur un dino ki prediction nahi banti

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
    experts.py     25 classic experts
    advanced.py    universal prediction (CTW), neural network, genetic-programming expert
    genetic.py     formula evolution (expression trees, crossover, mutation)
    formulas.py    formula discovery + holdout test
    ensemble.py    Fixed-Share Hedge, replay, scoring, post-mortem
    theorems.py    statistical findings
    stats.py       binomial / chi-square / normal
web/               dashboard
tests/             pytest
```

Test: `python -m pytest -q`
