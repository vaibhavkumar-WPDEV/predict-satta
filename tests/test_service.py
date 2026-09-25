import datetime as dt
import json

from satta import config, service, storage

from .conftest import synthetic_rows, write_results


def at(y, m, d, hh, mm=0):
    return dt.datetime(y, m, d, hh, mm, tzinfo=config.IST)


def test_cycle_locks_prediction_and_scores_it_later(data_dir):
    rows = synthetic_rows(60)  # 2026-01-01 .. 2026-03-01
    write_results(data_dir / "results.csv", rows)

    # morning of 2 March: predict 2 March, before 18:15
    service.cycle(fetch=False, now=at(2026, 3, 2, 9))
    preds = storage.load_predictions()
    frbd = [p for p in preds if p["market"] == "faridabad"]
    assert len(frbd) == 1 and frbd[0]["date"] == "2026-03-02"
    assert frbd[0]["late"] is False
    assert storage.verify_prediction(frbd[0])

    # running again does not create or change anything
    service.cycle(fetch=False, now=at(2026, 3, 2, 12))
    again = [p for p in storage.load_predictions() if p["market"] == "faridabad"]
    assert again == frbd

    # the result arrives, next cycle scores it and locks 3 March
    actual = frbd[0]["top10"][0][0]
    rows.append({"market": "faridabad", "date": "2026-03-02", "value": actual, "source": "t", "fetched_at": ""})
    write_results(data_dir / "results.csv", rows)
    service.cycle(fetch=False, now=at(2026, 3, 2, 19))
    dash = json.loads((data_dir / "dashboard.json").read_text())
    live = dash["markets"]["faridabad"]["live"]
    done = [r for r in live["rows"] if r["date"] == "2026-03-02"][0]
    assert done["status"] == "hit" and done["rank"] == 1 and done["verified"]
    assert live["summary"]["n"] == 1 and live["summary"]["hit1"]["hits"] == 1
    assert dash["markets"]["faridabad"]["next"]["date"] == "2026-03-03"


def test_tampering_is_detected(data_dir):
    write_results(data_dir / "results.csv", synthetic_rows(40))
    service.cycle(fetch=False, now=at(2026, 2, 10, 9))
    lines = storage.predictions_path().read_text().splitlines()
    p = json.loads(lines[1])
    p["top10"][0][0] = (p["top10"][0][0] + 1) % 100
    lines[1] = json.dumps(p)
    storage.predictions_path().write_text("\n".join(lines) + "\n")
    assert not storage.verify_prediction(storage.load_predictions()[1])


def test_late_prediction_is_flagged(data_dir):
    write_results(data_dir / "results.csv", synthetic_rows(40))  # last date 2026-02-09
    service.cycle(fetch=False, now=at(2026, 2, 10, 20))  # after 18:15
    frbd = [p for p in storage.load_predictions() if p["market"] == "faridabad"][0]
    assert frbd["date"] == "2026-02-10" and frbd["late"] is True


def test_learned_month_end_holiday_is_skipped(data_dir):
    rows = [r for r in synthetic_rows(150)  # 2026-01-01 .. 2026-05-30
            if not (r["market"] == "faridabad"
                    and (dt.date.fromisoformat(r["date"]) + dt.timedelta(days=1)).day == 1)]
    rows = [r for r in rows if r["date"] <= "2026-05-30"]
    write_results(data_dir / "results.csv", rows)
    service.cycle(fetch=False, now=at(2026, 5, 30, 20))  # after 30 May result
    frbd = [p for p in storage.load_predictions() if p["market"] == "faridabad"][0]
    assert frbd["date"] == "2026-06-01"  # 31 May is a learned holiday
    gali = [p for p in storage.load_predictions() if p["market"] == "gali"][0]
    assert gali["date"] == "2026-05-31"  # Gali had month-end results in this data


def test_holiday_moves_target_forward(data_dir):
    write_results(data_dir / "results.csv", synthetic_rows(40))  # last date 2026-02-09
    service.cycle(fetch=False, now=at(2026, 2, 13, 10))
    frbd = [p for p in storage.load_predictions() if p["market"] == "faridabad"][0]
    assert frbd["date"] == "2026-02-13"
