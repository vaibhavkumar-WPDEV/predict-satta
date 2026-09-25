"""Fetches Satta King charts from public result sites and merges them.

The parser does not depend on one site's exact HTML. It looks at every
<table> on the page and recognises three layouts:

  A. market columns   | DATE | DSWR | FRBD | GZBD | GALI |   (monthly chart)
  B. month columns    | DAY  | JAN | FEB | ... | DEC |      (yearly chart, one market)
  C. date + result    | 01-01-2026 | 45 |                   (list, one market)

Values that are not a two-digit number ("XX", "--", blank) are skipped.
Several sources are merged by majority vote, ties go to the higher priority.
"""

from __future__ import annotations

import calendar
import datetime as dt
import logging
import re
import time
from collections import Counter, defaultdict

import requests
from bs4 import BeautifulSoup

from . import config, storage

log = logging.getLogger("satta.scraper")

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
VALUE_RE = re.compile(r"^\D{0,3}(\d{2})\D{0,12}$")
FULL_DATE_RES = [
    (re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})"), ("y", "m", "d")),
    (re.compile(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})"), ("d", "m", "y")),
    (re.compile(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2})\b"), ("d", "m", "y2")),
]


# ------------------------------------------------------------ cell helpers

def _text(cell) -> str:
    return " ".join(cell.get_text(" ", strip=True).split())


def _letters(s: str) -> str:
    return re.sub(r"[^A-Z]", "", s.upper())


def market_of_header(text: str) -> str | None:
    """Map a header cell like 'FARIDABAD 06:15 PM' or 'FRBD' to a market key."""
    letters = _letters(text)
    if not letters:
        return None
    for key, meta in config.MARKETS.items():
        for alias in meta["aliases"]:
            if len(alias) >= 4 and alias in letters:
                return key
            if letters == alias:
                return key
    return None


def month_of_header(text: str) -> int | None:
    letters = _letters(text)
    if len(letters) < 3:
        return None
    for i, m in enumerate(MONTHS):
        if letters.startswith(m):
            return i + 1
    return None


def parse_value(text: str) -> int | None:
    text = text.strip()
    if not text or len(text) > 16:
        return None
    m = VALUE_RE.match(text)
    if not m:
        return None
    v = int(m.group(1))
    return v if 0 <= v <= 99 else None


def parse_date(text: str, year: int | None, month: int | None) -> dt.date | None:
    """Full dates are taken as-is; a bare day number needs year and month."""
    for rx, order in FULL_DATE_RES:
        m = rx.search(text)
        if m:
            parts = dict(zip(order, (int(g) for g in m.groups())))
            y = parts.get("y") or (2000 + parts["y2"])
            try:
                return dt.date(y, parts["m"], parts["d"])
            except ValueError:
                return None
    m = re.match(r"^\s*(\d{1,2})(?:st|nd|rd|th)?\b", text)
    if m and year and month:
        try:
            return dt.date(year, month, int(m.group(1)))
        except ValueError:
            return None
    return None


def _rows(table) -> list[list[str]]:
    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if cells:
            rows.append([_text(c) for c in cells])
    return rows


# ---------------------------------------------------------------- parsers

def parse_html(html: str, *, year: int | None = None, month: int | None = None,
               market: str | None = None) -> list[tuple[str, dt.date, int]]:
    """Return (market, date, value) triples found in any table of the page.

    year/month give context for bare day numbers; market fixes the market of
    single-market pages (yearly charts, date lists).
    """
    soup = BeautifulSoup(html, "html.parser")
    found: list[tuple[str, dt.date, int]] = []
    for table in soup.find_all("table"):
        rows = _rows(table)
        if len(rows) < 2:
            continue
        found.extend(_parse_table(rows, year, month, market))
    # de-duplicate, first occurrence wins
    seen: dict[tuple[str, dt.date], int] = {}
    for mk, d, v in found:
        seen.setdefault((mk, d), v)
    return [(mk, d, v) for (mk, d), v in seen.items()]


def _parse_table(rows, year, month, market):
    for hi, header in enumerate(rows[:4]):
        mcols = {i: market_of_header(c) for i, c in enumerate(header)}
        mcols = {i: mk for i, mk in mcols.items() if mk}
        if mcols and (market is None or market in mcols.values()):
            if market:
                mcols = {i: mk for i, mk in mcols.items() if mk == market}
            return _parse_market_columns(rows[hi + 1:], mcols, year, month)
        months = {i: month_of_header(c) for i, c in enumerate(header)}
        months = {i: mo for i, mo in months.items() if mo and i > 0}
        if len(months) >= 3 and market and year:
            return _parse_month_columns(rows[hi + 1:], months, year, market)
    if market:
        return _parse_date_list(rows, year, month, market)
    return []


def _parse_market_columns(rows, mcols, year, month):
    out = []
    for r in rows:
        if not r:
            continue
        d = parse_date(r[0], year, month)
        if d is None:
            continue
        for i, mk in mcols.items():
            if i < len(r):
                v = parse_value(r[i])
                if v is not None:
                    out.append((mk, d, v))
    return out


def _parse_month_columns(rows, months, year, market):
    out = []
    for r in rows:
        if not r:
            continue
        m = re.match(r"^\s*(\d{1,2})\b", r[0])
        if not m:
            continue
        day = int(m.group(1))
        for i, mo in months.items():
            if i < len(r):
                v = parse_value(r[i])
                if v is None:
                    continue
                try:
                    out.append((market, dt.date(year, mo, day), v))
                except ValueError:
                    pass
    return out


def _parse_date_list(rows, year, month, market):
    out = []
    for r in rows:
        if len(r) < 2:
            continue
        d = parse_date(r[0], year, month)
        if d is None or (year and d.year != year):
            continue
        v = parse_value(r[-1]) if len(r) == 2 else next(
            (x for x in (parse_value(c) for c in r[1:]) if x is not None), None)
        if v is not None:
            out.append((market, d, v))
    return out


# ------------------------------------------------------------------ fetch

def _get(url: str, session: requests.Session, tries: int = 2) -> str | None:
    for attempt in range(tries):
        try:
            resp = session.get(url, timeout=25, headers={"User-Agent": UA, "Accept-Language": "en-IN,en"})
            if resp.status_code == 200 and resp.text:
                return resp.text
            log.warning("GET %s -> HTTP %s", url, resp.status_code)
            if resp.status_code in (403, 404, 410):
                return None
        except requests.RequestException as exc:
            log.warning("GET %s failed: %s", url, exc)
        if attempt + 1 < tries:
            time.sleep(2)
    return None


def _months_between(start: dt.date, end: dt.date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def fetch_all(start: dt.date, end: dt.date, sources: list[dict] | None = None,
              session: requests.Session | None = None) -> tuple[list[tuple], list[dict]]:
    """Fetch every source for [start, end]. Returns (triples, status)."""
    sources = sources or config.load_sources()
    session = session or requests.Session()
    triples: list[tuple] = []
    status: list[dict] = []
    for src in sources:
        got = 0
        errors = 0
        if src["kind"] == "monthly":
            for y, m in _months_between(start, end):
                url = src["url"].format(year=y, mm=f"{m:02d}", m=m,
                                        month_name=calendar.month_name[m])
                html = _get(url, session)
                if html is None:
                    errors += 1
                    if errors >= 2 and got == 0:
                        break  # site down or blocked: don't hammer it
                    continue
                for mk, d, v in parse_html(html, year=y, month=m, market=src.get("market")):
                    if start <= d <= end and d.year == y and d.month == m:
                        triples.append((src["id"], src.get("priority", 9), mk, d, v))
                        got += 1
        else:
            for y in range(start.year, end.year + 1):
                html = _get(src["url"].format(year=y), session)
                if html is None:
                    errors += 1
                    continue
                for mk, d, v in parse_html(html, year=y, market=src.get("market")):
                    if start <= d <= end:
                        triples.append((src["id"], src.get("priority", 9), mk, d, v))
                        got += 1
        status.append({"source": src["id"], "values": got, "failed_pages": errors})
        log.info("source %s: %d values, %d failed pages", src["id"], got, errors)
    return triples, status


def merge(existing: list[dict], triples: list[tuple], fetched_at: str) -> tuple[list[dict], list[dict], int]:
    """Majority vote per (market, date). Returns (rows, conflicts, n_changed)."""
    votes: dict[tuple[str, str], list[tuple[int, int, str]]] = defaultdict(list)
    for sid, prio, mk, d, v in triples:
        votes[(mk, d.isoformat())].append((v, prio, sid))
    rows = {(r["market"], r["date"]): dict(r) for r in existing}
    conflicts = []
    changed = 0
    for key, vs in votes.items():
        counts = Counter(v for v, _, _ in vs)
        best_n = max(counts.values())
        tied = {v for v, c in counts.items() if c == best_n}
        winner = min((p, v) for v, p, _ in vs if v in tied)[1]
        if len(counts) > 1:
            conflicts.append({"market": key[0], "date": key[1],
                              "values": {sid: v for v, _, sid in vs}, "chosen": winner})
        old = rows.get(key)
        if old is None or int(old["value"]) != winner:
            changed += 1
            rows[key] = {"market": key[0], "date": key[1], "value": winner,
                         "source": "+".join(sorted({sid for v, _, sid in vs if v == winner})),
                         "fetched_at": fetched_at}
    return list(rows.values()), conflicts, changed


def sync(now: dt.datetime | None = None, full: bool = False) -> dict:
    """Download new results and write data/results.csv. Safe to call often."""
    now = now or config.now_ist()
    today = now.astimezone(config.IST).date()
    existing = storage.load_result_rows()
    if full or not existing:
        start = config.HISTORY_START
    else:
        # re-read the last ~40 days: catches late results and corrections
        start = max(config.HISTORY_START, today - dt.timedelta(days=40))
    triples, status = fetch_all(start, today)
    rows, conflicts, changed = merge(existing, triples, now.isoformat(timespec="seconds"))
    if changed:
        storage.save_result_rows(rows)
    return {"from": start.isoformat(), "to": today.isoformat(), "new_or_changed": changed,
            "sources": status, "conflicts": conflicts[-50:], "total_rows": len(rows)}


def import_csv(path: str) -> int:
    """Fallback: import a CSV with columns date,market,value (or date + one column per market)."""
    import csv

    triples = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            d = parse_date(r.get("date", ""), None, None)
            if d is None:
                continue
            if "market" in r and "value" in r:
                mk = market_of_header(r["market"]) or r["market"].strip().lower()
                v = parse_value(r["value"])
                if mk in config.MARKETS and v is not None:
                    triples.append(("import", 0, mk, d, v))
            else:
                for col, val in r.items():
                    mk = market_of_header(col or "")
                    v = parse_value(val or "")
                    if mk and v is not None:
                        triples.append(("import", 0, mk, d, v))
    rows, _, changed = merge(storage.load_result_rows(), triples,
                             config.now_ist().isoformat(timespec="seconds"))
    storage.save_result_rows(rows)
    return changed
