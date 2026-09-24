"""Service behaviour, with a stub model so no registry or network is involved.

The cases that matter here are the unhappy ones: what the service does before a
model is loaded, and what it does with a request that is the wrong shape.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlplatform import service


class StubModel:
    """Returns a fixed prediction per row, so assertions are about the service."""

    def predict(self, frame):
        return np.full(len(frame), 42.0)


VALID = {
    "season": 1,
    "mnth": 1,
    "hr": 8,
    "holiday": 0,
    "weekday": 1,
    "workingday": 1,
    "weathersit": 1,
    "temp": 0.24,
    "atemp": 0.28,
    "hum": 0.81,
    "windspeed": 0.0,
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    # The config objects are frozen, so swap in a copy rather than mutating one.
    from dataclasses import replace

    cfg = replace(
        service.CONFIG,
        service=replace(service.CONFIG.service, prediction_log=tmp_path / "pred.parquet"),
    )
    monkeypatch.setattr(service, "CONFIG", cfg)
    with TestClient(service.app) as c:
        yield c


@pytest.fixture
def loaded(client):
    service.STATE.model = StubModel()
    service.STATE.version = "test"
    yield client
    service.STATE.model = None
    service.STATE.version = "unloaded"


def test_liveness_is_true_even_without_a_model(client):
    """Liveness must not depend on the model, or a registry outage turns into a
    restart loop that cannot possibly fix itself."""
    service.STATE.model = None
    assert client.get("/health").status_code == 200


def test_readiness_is_false_without_a_model(client):
    service.STATE.model = None
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "no model"


def test_predicting_without_a_model_is_503_not_500(client):
    """503 tells a load balancer to try another instance. 500 does not."""
    service.STATE.model = None
    response = client.post("/predict", json={"observations": [VALID]})
    assert response.status_code == 503


def test_happy_path(loaded):
    response = loaded.post("/predict", json={"observations": [VALID, VALID]})
    assert response.status_code == 200
    body = response.json()
    assert body["predictions"] == [42.0, 42.0]
    assert body["model_version"] == "test"
    assert body["latency_ms"] >= 0


def test_out_of_range_field_is_rejected_before_the_model_sees_it(loaded):
    bad = dict(VALID, hr=99)
    assert loaded.post("/predict", json={"observations": [bad]}).status_code == 422


def test_empty_batch_is_rejected(loaded):
    assert loaded.post("/predict", json={"observations": []}).status_code == 422


def test_metrics_endpoint_exposes_the_counter(loaded):
    loaded.post("/predict", json={"observations": [VALID]})
    body = loaded.get("/metrics").text
    assert "predictions_total" in body
    assert "prediction_latency_seconds" in body


def test_predictions_are_logged_for_the_drift_monitor(loaded):
    """Without this log the monitor has nothing real to compare against."""
    import pandas as pd

    loaded.post("/predict", json={"observations": [VALID]})
    path = service.CONFIG.service.prediction_log
    assert path.exists()
    logged = pd.read_parquet(path)
    assert "prediction" in logged.columns
    assert len(logged) == 1
