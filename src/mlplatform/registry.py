"""MLflow tracking and the model registry.

Everything that talks to MLflow lives here. The rest of the platform works
without a tracking server running, which keeps the tests fast and means a
broken MLflow cannot stop you training locally.

Promotion uses a registry alias rather than a stage. Stages are deprecated in
MLflow 3, and an alias is simply a pointer you move, with no state machine
behind it to drift out of step with what is actually serving.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from .config import CONFIG, Config
from .train import TrainingRun

log = logging.getLogger(__name__)


def _mlflow(cfg: Config):
    # MLflow's HTTP client defaults to five retries with a 120 second timeout
    # each. Against an unreachable registry that is ten minutes of hanging
    # before anyone is told anything. Bounded here so a dead registry surfaces
    # as a failed readiness check in seconds instead.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", str(cfg.mlflow.request_timeout))
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", str(cfg.mlflow.max_retries))

    import mlflow

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    return mlflow


def _client(cfg: Config):
    _mlflow(cfg)
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=cfg.mlflow.tracking_uri)


def champion_metrics(cfg: Config | None = None) -> dict[str, float]:
    """Metrics logged with the version the champion alias currently points at.

    Returns an empty dict when there is no champion, or when MLflow is not
    reachable. A missing registry must not be able to block a training run, so
    every failure here is a warning rather than an exception.
    """
    cfg = cfg or CONFIG
    try:
        client = _client(cfg)
        version = client.get_model_version_by_alias(
            cfg.mlflow.registry_name, cfg.service.model_alias
        )
        run = client.get_run(version.run_id)
        return {k: float(v) for k, v in run.data.metrics.items()}
    except Exception as exc:
        log.warning("could not read champion metrics: %s", exc)
        return {}


def log_run(
    run: TrainingRun,
    cfg: Config | None = None,
    extra_params: dict[str, Any] | None = None,
) -> str | None:
    """Record the run, and move the champion alias only if the gate said so.

    Returns the MLflow run id, or None when tracking is unavailable. The caller
    treats None as "training succeeded, tracking did not", which is a warning
    rather than a failure.
    """
    cfg = cfg or CONFIG
    try:
        mlflow = _mlflow(cfg)
        mlflow.set_experiment(cfg.mlflow.experiment)

        with mlflow.start_run() as active:
            mlflow.log_params(
                {
                    "n_estimators": cfg.train.n_estimators,
                    "max_depth": cfg.train.max_depth,
                    "min_samples_leaf": cfg.train.min_samples_leaf,
                    "valid_fraction": cfg.data.valid_fraction,
                    "promotion_margin": cfg.train.promotion_margin,
                    "baseline_margin": cfg.train.baseline_margin,
                    **(extra_params or {}),
                }
            )
            mlflow.log_metrics(
                {
                    "valid_mae": run.valid_scores.mae,
                    "valid_rmse": run.valid_scores.rmse,
                    "valid_r2": run.valid_scores.r2,
                    "valid_rows": float(run.valid_scores.n_rows),
                    "baseline_mae": run.decision.baseline_mae,
                }
            )
            mlflow.set_tag("promoted", str(run.decision.promote))
            mlflow.set_tag("promotion_reason", run.decision.reason)

            # Every run is registered, so a refused challenger stays inspectable.
            # Only a promoted one gets the alias moved onto it.
            info = mlflow.sklearn.log_model(
                run.pipeline,
                name="model",
                signature=_signature(run),
                registered_model_name=cfg.mlflow.registry_name,
                skops_trusted_types=list(cfg.mlflow.trusted_types),
            )

            if run.decision.promote:
                _point_alias_at(info, cfg)

            return active.info.run_id
    except Exception as exc:
        log.warning("MLflow logging skipped: %s", exc)
        return None


def _signature(run: TrainingRun):
    """Pin the input schema to the model so a wrong-shaped request fails at the door."""
    from mlflow.models.signature import infer_signature

    sample = run.reference.head(100)
    return infer_signature(sample, run.pipeline.predict(sample))


def _point_alias_at(model_info, cfg: Config) -> None:
    """Move the champion alias onto the version just logged.

    The version number comes back on the log_model result, so there is no
    "find the newest version" race if two runs finish at once.
    """
    client = _client(cfg)
    version = getattr(model_info, "registered_model_version", None)
    if version is None:
        versions = client.search_model_versions(f"name='{cfg.mlflow.registry_name}'")
        if not versions:
            log.warning("nothing registered, alias not moved")
            return
        version = max(versions, key=lambda v: int(v.version)).version

    client.set_registered_model_alias(
        cfg.mlflow.registry_name, cfg.service.model_alias, str(version)
    )
    log.info(
        "alias '%s' now points at %s version %s",
        cfg.service.model_alias,
        cfg.mlflow.registry_name,
        version,
    )


def champion_uri(cfg: Config | None = None) -> str:
    cfg = cfg or CONFIG
    return f"models:/{cfg.mlflow.registry_name}@{cfg.service.model_alias}"


def load_champion(cfg: Config | None = None):
    """Load the serving model.

    Raises when there is none, because a service with no model should fail its
    readiness check rather than answer with nonsense.
    """
    cfg = cfg or CONFIG
    mlflow = _mlflow(cfg)
    return mlflow.sklearn.load_model(champion_uri(cfg))


def champion_version(cfg: Config | None = None) -> str:
    """The registry version currently behind the alias, for the service to report."""
    cfg = cfg or CONFIG
    try:
        client = _client(cfg)
        version = client.get_model_version_by_alias(
            cfg.mlflow.registry_name, cfg.service.model_alias
        )
        return str(version.version)
    except Exception:
        return "unknown"


def load_reference(cfg: Config | None = None) -> pd.DataFrame | None:
    """The training window the champion saw, for the drift monitor to compare against."""
    cfg = cfg or CONFIG
    from .data import period_frames
    from .features import feature_frame

    try:
        return feature_frame(period_frames(cfg.data).reference, cfg)
    except Exception as exc:
        log.warning("reference window unavailable: %s", exc)
        return None
