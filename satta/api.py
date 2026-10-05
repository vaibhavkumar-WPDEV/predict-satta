"""FastAPI backend + background scheduler + static frontend."""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, service, storage

log = logging.getLogger("satta.api")
_stop = threading.Event()
_state = {"last_cycle": None, "last_error": None, "running": False}


def _run_pattern() -> None:
    """The Pattern Engine and Engine 5 run after engine 4.0; a failure never touches engine 4.0."""
    from .pattern import service as pattern

    for fl in pattern.FLAVORS.values():
        try:
            pattern.cycle(fl=fl)
        except Exception:
            log.exception("%s failed", fl.kind)


def _run_cycle(fetch: bool = True) -> dict:
    _state["running"] = True
    try:
        res = service.cycle(fetch=fetch)
        _state["last_cycle"] = res
        _state["last_error"] = None
        _run_pattern()
        return res
    except Exception as exc:
        log.exception("cycle failed")
        _state["last_error"] = str(exc)
        raise
    finally:
        _state["running"] = False


def _scheduler():
    while not _stop.is_set():
        try:
            _run_cycle(fetch=True)
        except Exception:
            pass
        _stop.wait(config.CYCLE_MINUTES * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = None
    if os.environ.get("SATTA_NO_SCHEDULER") != "1":
        t = threading.Thread(target=_scheduler, name="satta-scheduler", daemon=True)
        t.start()
    yield
    _stop.set()


app = FastAPI(title="Satta Predictor", version=config.ENGINE_VERSION, lifespan=lifespan)


@app.get("/api/dashboard")
async def dashboard():
    dash = service.load_dashboard()
    if dash is None:
        await run_in_threadpool(service.cycle, False)
        dash = service.load_dashboard()
    dash["server"] = {**_state, "cycle_minutes": config.CYCLE_MINUTES}
    return JSONResponse(dash)


@app.post("/api/cycle")
async def run_cycle(fetch: bool = True):
    if _state["running"]:
        raise HTTPException(409, "Ek cycle pehle se chal raha hai, thoda ruko.")
    return await run_in_threadpool(_run_cycle, fetch)


@app.get("/api/backtest")
async def backtest(market: str = config.PRIMARY_MARKET, days: int = 7):
    if market not in config.MARKETS:
        raise HTTPException(404, "unknown market")
    try:
        rows, summary = await run_in_threadpool(service.backtest, market, max(1, min(days, 365)))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"market": market, "days": days, "summary": service._clean(summary), "rows": service._clean(rows)}


@app.get("/api/predictions")
async def predictions():
    preds = storage.load_predictions()
    for p in preds:
        p["verified"] = storage.verify_prediction(p)
        p.pop("dist", None)
    return preds


@app.get("/api/results.csv")
async def results_csv():
    path = storage.results_path()
    if not path.exists():
        raise HTTPException(404, "abhi koi result nahi")
    return FileResponse(path, media_type="text/csv", filename="satta-results.csv")


UPGRADED = ("pattern", "engine5")


@app.get("/api/{kind}/dashboard")
async def upgraded_dashboard(kind: str):
    path = config.DATA_DIR / kind / "dashboard.json"
    if kind not in UPGRADED or not path.exists():
        raise HTTPException(404, "Yeh engine abhi chala nahi")
    return FileResponse(path, media_type="application/json")


@app.get("/api/{kind}/prediction_history.csv")
async def upgraded_history_csv(kind: str):
    path = config.DATA_DIR / kind / "prediction_history.csv"
    if kind not in UPGRADED or not path.exists():
        raise HTTPException(404, "Is engine ki abhi koi locked prediction nahi")
    return FileResponse(path, media_type="text/csv", filename=f"{kind}-prediction-history.csv")


@app.get("/api/prediction_history.csv")
async def prediction_history_csv():
    path = config.DATA_DIR / "prediction_history.csv"
    if not path.exists():
        raise HTTPException(404, "abhi koi locked prediction nahi")
    return FileResponse(path, media_type="text/csv", filename="satta-prediction-history.csv")


@app.get("/api/health")
async def health():
    return {"ok": True, **_state}


# Frontend (must be last so /api/* wins)
app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")
