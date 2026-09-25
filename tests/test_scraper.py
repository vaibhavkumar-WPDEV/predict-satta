import datetime as dt

from satta import scraper

MONTHLY = """
<html><body>
<table><tr><td>Today</td><td>45</td></tr></table>
<table>
  <tr><th>2026</th><th>DSWR</th><th>FRBD</th><th>GZBD</th><th>GALI</th></tr>
  <tr><td>01</td><td>12</td><td>34</td><td>56</td><td>78</td></tr>
  <tr><td>02</td><td>05</td><td>XX</td><td>--</td><td>90</td></tr>
  <tr><td>03</td><td></td><td>07</td><td>99</td><td>00</td></tr>
</table></body></html>
"""

YEARLY = """
<table>
  <thead><tr><th>Day</th><th>Jan</th><th>Feb</th><th>Mar</th><th>Apr</th></tr></thead>
  <tbody>
    <tr><td>1</td><td>11</td><td>21</td><td>31</td><td>XX</td></tr>
    <tr><td>30</td><td>41</td><td></td><td>51</td><td>61</td></tr>
    <tr><td>31</td><td>71</td><td>--</td><td>81</td><td></td></tr>
  </tbody>
</table>
"""

LONG_HEADERS = """
<table>
  <tr><td>Date</td><td>DESAWAR 05:00 AM</td><td>FARIDABAD 06:15 PM</td><td>GHAZIABAD 09:30 PM</td><td>GALI 11:30 PM</td></tr>
  <tr><td>15-02-2026</td><td>10</td><td>20</td><td>30</td><td>40</td></tr>
</table>
"""


def test_monthly_market_columns():
    got = {(m, d.day): v for m, d, v in scraper.parse_html(MONTHLY, year=2026, month=3)}
    assert got[("disawar", 1)] == 12
    assert got[("faridabad", 1)] == 34
    assert got[("gali", 1)] == 78
    assert got[("disawar", 2)] == 5
    assert ("faridabad", 2) not in got  # XX
    assert ("ghaziabad", 2) not in got  # --
    assert got[("faridabad", 3)] == 7
    assert got[("gali", 3)] == 0
    assert all(d.month == 3 for _, d, _ in scraper.parse_html(MONTHLY, year=2026, month=3))


def test_yearly_month_columns():
    got = {d: v for m, d, v in scraper.parse_html(YEARLY, year=2026, market="faridabad")}
    assert got[dt.date(2026, 1, 1)] == 11
    assert got[dt.date(2026, 2, 1)] == 21
    assert got[dt.date(2026, 4, 30)] == 61
    assert got[dt.date(2026, 3, 31)] == 81
    assert dt.date(2026, 4, 1) not in got
    assert len(got) == 8  # Feb 30/31 and blanks skipped


def test_long_headers_and_full_dates():
    got = scraper.parse_html(LONG_HEADERS)
    assert ("faridabad", dt.date(2026, 2, 15), 20) in got
    assert ("disawar", dt.date(2026, 2, 15), 10) in got
    assert ("ghaziabad", dt.date(2026, 2, 15), 30) in got
    assert ("gali", dt.date(2026, 2, 15), 40) in got


def test_single_market_page_ignores_other_columns():
    got = scraper.parse_html(LONG_HEADERS, market="faridabad")
    assert got == [("faridabad", dt.date(2026, 2, 15), 20)]


def test_merge_majority_vote():
    d = dt.date(2026, 5, 1)
    triples = [("a", 1, "faridabad", d, 11), ("b", 2, "faridabad", d, 22), ("c", 3, "faridabad", d, 22)]
    rows, conflicts, changed = scraper.merge([], triples, "now")
    assert rows[0]["value"] == 22 and changed == 1 and len(conflicts) == 1
    # tie -> higher priority (lower number) wins
    rows, _, _ = scraper.merge([], triples[:2], "now")
    assert rows[0]["value"] == 11
