"""Engine changelog with the walk-forward Top-10 hit rate each version reached.

Numbers are the backtest over the history the version used (every day predicted
only from earlier days). Earlier versions used less history, so their rows are
on different spans; within a row all four markets share the same span.
"""

VERSIONS = [
    {"version": "1.0", "date": "2026-09-25", "span": "2026 (~230 din)",
     "change": "20 models, Fixed-Share Hedge ensemble, formula search, walk-forward backtest, hash-locked predictions",
     "top10": {"disawar": 9.2, "faridabad": 9.5, "ghaziabad": 8.3, "gali": 8.7}},
    {"version": "2.0", "date": "2026-09-26", "span": "2025–26 (~583 din)",
     "change": "Aryabhata, Fibonacci, Vedic/Tesla, Einstein models; shuffle test; learned vs equal weights",
     "top10": {"disawar": 10.0, "faridabad": 8.7, "ghaziabad": 11.1, "gali": 8.6}},
    {"version": "2.1", "date": "2026-09-26", "span": "2025–26 (~583 din)",
     "change": "Universal prediction (CTW), neural network, meta-learning of eta/alpha",
     "top10": {"disawar": 9.8, "faridabad": 8.4, "ghaziabad": 9.6, "gali": 8.1}},
    {"version": "3.0", "date": "2026-09-26", "span": "2021–26 (~1,996 din)",
     "change": "5 saal ka data, genetic programming, linear+geometric merge",
     "top10": {"disawar": 9.8, "faridabad": 8.9, "ghaziabad": 9.3, "gali": 8.5}},
    {"version": "3.1", "date": "2026-09-26", "span": "2021–26 (~1,996 din)",
     "change": "Same-day market connection, miss-correction layer",
     "top10": {"disawar": 8.7, "faridabad": 9.0, "ghaziabad": 9.3, "gali": 9.1}},
    {"version": "3.2", "date": "2026-09-26", "span": "2021–26 (~1,996 din)",
     "change": "Hot pool (number ghoom ke aata hai) + Top-10 selector — pehla asli sudhaar",
     "top10": {"disawar": 11.4, "faridabad": 9.5, "ghaziabad": 10.0, "gali": 10.7}},
    {"version": "3.3", "date": "2026-09-26", "span": "2021–26 (~1,996 din)",
     "change": "Signal quality, EDGE/NO-EDGE decision, calibrated probability, numerology+tithi layer, "
               "cyclical time features, arrival-time window, 30-din Monte Carlo",
     "top10": None},  # filled in from the live dashboard (current backtest)
]
