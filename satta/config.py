"""Static configuration: markets, result times, data sources, paths."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SATTA_DATA_DIR", ROOT / "data"))
WEB_DIR = ROOT / "web"

# History that is fetched on the first run. Everything after this date is
# downloaded automatically; the models train on all of it.
HISTORY_START = dt.date.fromisoformat(os.environ.get("SATTA_HISTORY_START", "2021-01-01"))

ENGINE_VERSION = "3.3"

# Result times are approximate (IST). They decide when a prediction stops
# counting as "live" (made before the result) and when the scheduler looks for
# a new result.
MARKETS: dict[str, dict] = {
    "disawar": {
        "name": "Disawar",
        "short": "DSWR",
        "aliases": ["DSWR", "DISAWAR", "DESAWAR", "DISAWER", "DESAWER"],
        "result_time": "05:00",
    },
    "faridabad": {
        "name": "Faridabad",
        "short": "FRBD",
        "aliases": ["FRBD", "FARIDABAD", "FBD", "FARIDABAAD"],
        "result_time": "18:15",
    },
    "ghaziabad": {
        "name": "Ghaziabad",
        "short": "GZBD",
        "aliases": ["GZBD", "GHAZIABAD", "GAZIABAD", "GAZIYABAD", "GHAZIYABAD", "GZB"],
        "result_time": "21:30",
    },
    "gali": {
        "name": "Gali",
        "short": "GALI",
        "aliases": ["GALI"],
        "result_time": "23:30",
    },
}
MARKET_KEYS = list(MARKETS)
PRIMARY_MARKET = "faridabad"

# Public chart pages. "monthly" pages hold all four markets for one month,
# "yearly" pages hold one market for a whole year (day rows x month columns).
# The parser is layout-agnostic, so a new source usually needs only its URL.
# Extra/overriding sources can be put in data/sources.json (same shape).
DEFAULT_SOURCES: list[dict] = [
    {
        "id": "satta-king-fast",
        "kind": "monthly",
        "url": "https://satta-king-fast.com/chart.php?ResultFor={month_name}-{year}&month={mm}&year={year}",
        "priority": 1,
    },
    {
        "id": "satta-company",
        "kind": "yearly",
        "market": "faridabad",
        "url": "https://satta-company.com/faridabad-chart-{year}.php",
        "priority": 2,
    },
    {
        "id": "sattacom",
        "kind": "yearly",
        "market": "faridabad",
        "url": "https://www.sattacom.in/satta-king-record-chart-faridabad-{year}.php",
        "priority": 3,
    },
    {
        "id": "satta-xpress",
        "kind": "yearly",
        "market": "faridabad",
        "url": "https://satta-xpress.com/faridabad-chart-{year}.php",
        "priority": 4,
    },
]

# Scheduler: how often the server re-checks for new results (minutes).
CYCLE_MINUTES = int(os.environ.get("SATTA_CYCLE_MINUTES", "15"))

# Walk-forward replay starts after this many draws of a market.
MIN_TRAIN = 30


def load_sources() -> list[dict]:
    path = DATA_DIR / "sources.json"
    if path.exists():
        try:
            extra = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(extra, list) and extra:
                return extra
        except (OSError, ValueError):
            pass
    return DEFAULT_SOURCES


def result_datetime(market: str, date: dt.date) -> dt.datetime:
    hh, mm = (int(x) for x in MARKETS[market]["result_time"].split(":"))
    return dt.datetime.combine(date, dt.time(hh, mm), tzinfo=IST)


def now_ist() -> dt.datetime:
    return dt.datetime.now(IST)
