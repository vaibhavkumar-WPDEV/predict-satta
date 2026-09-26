"""Engine changelog with the walk-forward Top-10 hit rate each version reached.

Every version's code (from git history) was re-run on the same 2021-26
results, walk-forward (every day predicted only from earlier days), so the
rows are directly comparable. "all" = the four markets together (7,987 days).
"""

VERSIONS = [
    {"version": "1.0", "date": "2026-09-25", "span": "2021–26, sabhi versions same data (1,996 din/market)",
     "change": "20 models, Fixed-Share Hedge ensemble, formula search, walk-forward backtest, hash-locked predictions",
     "top10": {"disawar": 10.4, "faridabad": 8.9, "ghaziabad": 9.5, "gali": 9.4, "all": 9.55}},
    {"version": "2.0", "date": "2026-09-26", "span": "2021–26, sabhi versions same data (1,996 din/market)",
     "change": "Aryabhata, Fibonacci, Vedic/Tesla, Einstein models; shuffle test; learned vs equal weights",
     "top10": {"disawar": 10.0, "faridabad": 9.0, "ghaziabad": 10.1, "gali": 9.1, "all": 9.52}},
    {"version": "2.1", "date": "2026-09-26", "span": "2021–26, sabhi versions same data (1,996 din/market)",
     "change": "Universal prediction (CTW), neural network, meta-learning of eta/alpha",
     "top10": {"disawar": 9.7, "faridabad": 8.8, "ghaziabad": 9.7, "gali": 8.7, "all": 9.23}},
    {"version": "3.0", "date": "2026-09-26", "span": "2021–26, sabhi versions same data (1,996 din/market)",
     "change": "5 saal ka data, genetic programming, linear+geometric merge",
     "top10": {"disawar": 9.8, "faridabad": 8.9, "ghaziabad": 9.3, "gali": 8.5, "all": 9.11}},
    {"version": "3.1", "date": "2026-09-26", "span": "2021–26, sabhi versions same data (1,996 din/market)",
     "change": "Same-day market connection, miss-correction layer",
     "top10": {"disawar": 8.7, "faridabad": 9.0, "ghaziabad": 9.3, "gali": 9.1, "all": 9.03}},
    {"version": "3.2", "date": "2026-09-26", "span": "2021–26, sabhi versions same data (1,996 din/market)",
     "change": "Hot pool (number ghoom ke aata hai) + Top-10 selector — pehla asli sudhaar",
     "top10": {"disawar": 11.4, "faridabad": 9.5, "ghaziabad": 10.0, "gali": 10.7, "all": 10.42}},
    {"version": "3.3", "date": "2026-09-26", "span": "2021–26, sabhi versions same data (1,996 din/market)",
     "change": "Signal quality, EDGE/NO-EDGE decision, calibrated probability, numerology+tithi layer, "
               "cyclical time features, arrival-time window, 30-din Monte Carlo",
     "top10": {"disawar": 11.4, "faridabad": 10.0, "ghaziabad": 10.0, "gali": 10.8, "all": 10.54}},
]
