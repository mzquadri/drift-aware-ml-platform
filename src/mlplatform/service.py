"""The prediction service.

Three things it does that a bare `@app.post("/predict")` does not:

  * separates liveness from readiness, so an orchestrator restarts a hung
    process but only routes traffic once a model is actually loaded;
  * exports Prometheus metrics including the prediction distribution, because
    "the model is up" and "the model is behaving" are different questions;
  * appends every request to a parquet log, which is what gives the drift
    monitor real serving traffic to compare against the training window.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from .config import CONFIG
from .features import SchemaError, feature_frame

log = logging.getLogger(__name__)

PREDICTIONS = Counter("predictions_total", "Prediction requests served", ["outcome"])
LATENCY = Histogram(
    "prediction_latency_seconds",
    "Wall clock time to answer a prediction request",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
PREDICTED_VALUE = Histogram(
    "predicted_demand",
    "Distribution of predicted hourly demand",
    buckets=(0, 25, 50, 100, 200, 400, 600, 800, 1000),
)
MODEL_READY = Gauge("model_ready", "1 when a champion model is loaded, 0 otherwise")
MODEL_VERSION = Gauge("model_version_info", "Loaded model version", ["version"])


class Observation(BaseModel):
    """One hour to score. Field names match the dataset so there is no mapping layer."""

    season: int = Field(ge=1, le=4)
    mnth: int = Field(ge=1, le=12)
    hr: int = Field(ge=0, le=23)
    holiday: int = Field(ge=0, le=1)
    weekday: int = Field(ge=0, le=6)
    workingday: int = Field(ge=0, le=1)
    weathersit: int = Field(ge=1, le=4)
    temp: float = Field(ge=0.0, le=1.0)
    atemp: float = Field(ge=0.0, le=1.0)
    hum: float = Field(ge=0.0, le=1.0)
    windspeed: float = Field(ge=0.0, le=1.0)


class PredictRequest(BaseModel):
    observations: list[Observation] = Field(min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    predictions: list[float]
    model_version: str
    latency_ms: float


class _State:
    """Holds the loaded model. Kept in one object so readiness has a single source."""

    def __init__(self) -> None:
        self.model: Any = None
        self.version: str = "unloaded"

    @property
    def ready(self) -> bool:
        return self.model is not None


STATE = _State()


def _load_model() -> None:
    from .registry import load_champion

    try:
        STATE.model = load_champion(CONFIG)
        STATE.version = CONFIG.service.model_stage
        MODEL_READY.set(1)
        MODEL_VERSION.labels(version=STATE.version).set(1)
        log.info("champion model loaded from the registry")
    except Exception as exc:
        STATE.model = None
        MODEL_READY.set(0)
        log.error("no champion model available: %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="Drift-aware demand service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness. The process is running and can answer. Says nothing about the model."""
    return {"status": "alive"}


@app.get("/ready")
def ready(response: Response) -> dict[str, Any]:
    """Readiness. Only true once a model is loaded, so traffic is not routed early."""
    if not STATE.ready:
        response.status_code = 503
        return {"status": "no model", "model_version": STATE.version}
    return {"status": "ready", "model_version": STATE.version}


@app.post("/reload", status_code=202)
def reload_model() -> dict[str, str]:
    """Pick up a newly promoted champion without restarting the container."""
    _load_model()
    return {"status": "reloaded" if STATE.ready else "failed", "model_version": STATE.version}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if not STATE.ready:
        PREDICTIONS.labels(outcome="unavailable").inc()
        raise HTTPException(status_code=503, detail="no champion model loaded")

    started = time.perf_counter()
    frame = pd.DataFrame([o.model_dump() for o in request.observations])

    try:
        features = feature_frame(frame, CONFIG)
    except SchemaError as exc:
        PREDICTIONS.labels(outcome="bad_request").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        values = [float(v) for v in STATE.model.predict(features)]
    except Exception as exc:
        PREDICTIONS.labels(outcome="error").inc()
        log.exception("prediction failed")
        raise HTTPException(status_code=500, detail="prediction failed") from exc

    elapsed = time.perf_counter() - started
    LATENCY.observe(elapsed)
    PREDICTIONS.labels(outcome="ok").inc(len(values))
    for v in values:
        PREDICTED_VALUE.observe(v)

    _append_log(frame, values)

    return PredictResponse(
        predictions=values,
        model_version=STATE.version,
        latency_ms=round(elapsed * 1000, 3),
    )


def _append_log(frame: pd.DataFrame, predictions: list[float]) -> None:
    """Persist what we were asked and what we answered.

    Best effort on purpose: a full disk should degrade monitoring, not take the
    service down. The failure is logged so it does not pass unnoticed.
    """
    path = CONFIG.service.prediction_log
    try:
        record = frame.copy()
        record["prediction"] = predictions
        record["logged_at"] = pd.Timestamp.utcnow()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            record = pd.concat([pd.read_parquet(path), record], ignore_index=True)
        record.to_parquet(path, index=False)
    except Exception as exc:
        log.warning("prediction log write failed, monitoring will be thin: %s", exc)
