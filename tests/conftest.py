import csv
import datetime as dt
import random

import pytest

from satta import config


def write_results(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["market", "date", "value", "source", "fetched_at"])
        w.writeheader()
        w.writerows(rows)


def synthetic_rows(days=150, seed=1, planted=False, start=dt.date(2026, 1, 1)):
    """Random results; with planted=True Faridabad = (Gali kal + 37) mod 100 on 40 % of days."""
    rnd = random.Random(seed)
    rows, prev_gali = [], None
    for k in range(days):
        d = start + dt.timedelta(days=k)
        vals = {m: rnd.randint(0, 99) for m in ("disawar", "ghaziabad", "gali")}
        if planted and prev_gali is not None and rnd.random() < 0.4:
            vals["faridabad"] = (prev_gali + 37) % 100
        else:
            vals["faridabad"] = rnd.randint(0, 99)
        for m, v in vals.items():
            rows.append({"market": m, "date": d.isoformat(), "value": v, "source": "test", "fetched_at": ""})
        prev_gali = vals["gali"]
    return rows


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return tmp_path
