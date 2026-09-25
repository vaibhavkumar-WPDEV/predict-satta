"""Plain-text storage (git friendly, so every commit is a timestamped proof).

data/results.csv       one row per (market, date) result
data/predictions.jsonl append-only log of locked live predictions
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

from . import config

RESULT_FIELDS = ["market", "date", "value", "source", "fetched_at"]


def _dir() -> Path:
    d = Path(config.DATA_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_path() -> Path:
    return _dir() / "results.csv"


def predictions_path() -> Path:
    return _dir() / "predictions.jsonl"


def cache_dir() -> Path:
    d = _dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- results

def load_result_rows() -> list[dict]:
    path = results_path()
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                value = int(r["value"])
                dt.date.fromisoformat(r["date"])
            except (KeyError, TypeError, ValueError):
                continue
            if r.get("market") in config.MARKETS and 0 <= value <= 99:
                r["value"] = value
                rows.append(r)
    return rows


def save_result_rows(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (r["date"], config.MARKET_KEYS.index(r["market"])))
    tmp = results_path().with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in RESULT_FIELDS})
    tmp.replace(results_path())


def load_table() -> dict[str, dict[dt.date, int]]:
    """{market: {date: value}} for every market (empty dict if no data)."""
    table: dict[str, dict[dt.date, int]] = {m: {} for m in config.MARKET_KEYS}
    for r in load_result_rows():
        table[r["market"]][dt.date.fromisoformat(r["date"])] = int(r["value"])
    return table


# ------------------------------------------------------------ predictions

def load_predictions() -> list[dict]:
    path = predictions_path()
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def append_prediction(pred: dict) -> None:
    with predictions_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(pred, separators=(",", ":"), ensure_ascii=False) + "\n")


def prediction_hash(pred: dict) -> str:
    """SHA-256 over the parts that must not change after locking."""
    core = {k: pred[k] for k in ("market", "date", "created_at", "top10", "dist")}
    blob = json.dumps(core, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def verify_prediction(pred: dict) -> bool:
    return pred.get("hash") == prediction_hash(pred)
