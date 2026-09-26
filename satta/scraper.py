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
import os
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
    """Map a header cell like 'FARIDABAD 06:15 PM' or 'FRBD' to a market key.

    Captions such as 'Chart for Gali, Desawar, Ghaziabad and Faridabad' name
    several markets and are not column headers, so they map to None.
    """
    if len(text) > 40:
        return None
    letters = _letters(text)
    if not letters:
        return None
    hits = set()
    for key, meta in config.MARKETS.items():
        for alias in meta["aliases"]:
            if (len(alias) >= 4 and alias in letters) or letters == alias:
                hits.add(key)
    return hits.pop() if len(hits) == 1 else None


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
    for hi, header in enumerate(rows[:6]):
        # column 0 is the date column, never a market
        mcols = {i: market_of_header(c) for i, c in enumerate(header) if i > 0}
        mcols = {i: mk for i, mk in mcols.items() if mk}
        enough = len(set(mcols.values())) >= 2 if market is None else market in mcols.values()
        if mcols and enough:
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

def _get(url: str, session: requests.Session, tries: int = 3) -> tuple[str | None, str]:
    """(html, status) with status "ok", "missing" (page does not exist) or "error".

    Busy sites often answer a first request with 403/5xx or time out, so errors
    are retried with a growing pause.
    """
    for attempt in range(tries):
        try:
            resp = session.get(url, timeout=30, headers={"User-Agent": UA, "Accept-Language": "en-IN,en"})
            if resp.status_code == 200 and resp.text:
                return resp.text, "ok"
            log.warning("GET %s -> HTTP %s", url, resp.status_code)
            if resp.status_code in (404, 410):
                return None, "missing"
        except requests.RequestException as exc:
            log.warning("GET %s failed: %s", url, exc)
        if attempt + 1 < tries:
            time.sleep(3 * (attempt + 1))
    return None, "error"


def _months_between(start: dt.date, end: dt.date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def sanitize(found: list[tuple[str, dt.date, int]], now: dt.datetime) -> tuple[list, dict]:
    """Drop values that cannot be real results.

    * a column whose values mostly equal the day of month is a mis-read date column
    * a result dated in the future, or before its declaration time today
    """
    dropped = {"day_number_columns": 0, "not_declared_yet": 0}
    by_market: dict[str, list] = defaultdict(list)
    for t in found:
        by_market[t[0]].append(t)
    out = []
    for mk, items in by_market.items():
        same_as_day = sum(1 for _, d, v in items if v == d.day)
        if len(items) >= 5 and same_as_day >= 0.5 * len(items):
            dropped["day_number_columns"] += len(items)
            continue
        for t in items:
            if now < config.result_datetime(mk, t[1]):
                dropped["not_declared_yet"] += 1
            else:
                out.append(t)
    return out, dropped


def _dump_html(src_id: str, html: str) -> None:
    if os.environ.get("SATTA_DUMP_HTML") == "1":
        d = storage.raw_dir() / "html"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{src_id}.html").write_text(html, encoding="utf-8")


def fetch_all(start: dt.date, end: dt.date, sources: list[dict] | None = None,
              session: requests.Session | None = None,
              now: dt.datetime | None = None) -> tuple[list[tuple], list[dict]]:
    """Fetch every source for [start, end]. Returns (triples, status)."""
    sources = sources or config.load_sources()
    session = session or requests.Session()
    now = now or config.now_ist()
    triples: list[tuple] = []
    status: list[dict] = []
    for src in sources:
        got = 0
        errors = 0
        missing = 0
        dropped = Counter()
        pages = []
        if src["kind"] == "monthly":
            for y, m in _months_between(start, end):
                pages.append(((y, m), src["url"].format(year=y, mm=f"{m:02d}", m=m,
                                                         month_name=calendar.month_name[m])))
        else:
            pages = [((y, None), src["url"].format(year=y)) for y in range(start.year, end.year + 1)]
        for (y, m), url in pages:
            html, state = _get(url, session)
            if html is None:
                if state == "missing":
                    missing += 1  # e.g. a year the site has no chart for
                    continue
                errors += 1
                if errors >= 2 and got == 0:
                    break  # site down or blocked: don't hammer it
                continue
            _dump_html(src["id"], html)
            found = [(mk, d, v) for mk, d, v in parse_html(html, year=y, month=m, market=src.get("market"))
                     if start <= d <= end and d.year == y and (m is None or d.month == m)]
            found, drop = sanitize(found, now)
            dropped.update(drop)
            for mk, d, v in found:
                triples.append((src["id"], src.get("priority", 9), mk, d, v))
                got += 1
        status.append({"source": src["id"], "values": got, "failed_pages": errors,
                       "missing_pages": missing, "dropped": dict(dropped)})
        log.info("source %s: %d values, %d failed pages, dropped %s", src["id"], got, errors, dict(dropped))
    return triples, status


def agreement(triples: list[tuple]) -> list[dict]:
    """How often two sources agree on the same (market, date), and with a ±1 day shift.

    A pair that agrees much better when shifted has one chart off by a day.
    """
    series: dict[tuple[str, str], dict[dt.date, int]] = defaultdict(dict)
    for sid, _, mk, d, v in triples:
        series[(sid, mk)][d] = v
    out = []
    keys = sorted(series)
    for i, (sa, ma) in enumerate(keys):
        for sb, mb in keys[i + 1:]:
            if ma != mb:
                continue
            a, b = series[(sa, ma)], series[(sb, mb)]
            row = {"market": ma, "a": sa, "b": sb}
            for name, shift in (("same_day", 0), ("b_is_next_day", 1), ("b_is_prev_day", -1)):
                common = [d for d in a if d + dt.timedelta(days=shift) in b]
                agree = sum(1 for d in common if a[d] == b[d + dt.timedelta(days=shift)])
                row[name] = {"n": len(common), "agree": agree}
            out.append(row)
    return out


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
    per_market = Counter(r["market"] for r in existing)
    expected = (today - config.HISTORY_START).days * 0.8
    state = storage.load_sync_state()
    last_full = state.get("last_full")
    stale = (state.get("history_start") != config.HISTORY_START.isoformat() or not last_full
             or now - dt.datetime.fromisoformat(last_full) > dt.timedelta(hours=24))
    gaps = any(per_market[m] < expected for m in config.MARKET_KEYS)
    if full or not existing or (gaps and stale):
        # first run, a longer history was asked for, or gaps (retried once a day)
        start = config.HISTORY_START
        state.update(history_start=config.HISTORY_START.isoformat(),
                     last_full=now.isoformat(timespec="seconds"))
        storage.save_sync_state(state)
    else:
        # re-read the last ~40 days: catches late results and corrections
        start = max(config.HISTORY_START, today - dt.timedelta(days=40))
    triples, status = fetch_all(start, today, now=now)
    storage.save_raw(triples)
    rows, conflicts, changed = merge(existing, triples, now.isoformat(timespec="seconds"))
    if changed:
        storage.save_result_rows(rows)
    return {"from": start.isoformat(), "to": today.isoformat(), "new_or_changed": changed,
            "sources": status, "agreement": agreement(triples),
            "conflicts": conflicts[-50:], "total_rows": len(rows)}


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
