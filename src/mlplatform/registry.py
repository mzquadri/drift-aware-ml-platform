"""MLflow tracking and the model registry.

Everything that talks to MLflow lives here. The rest of the platform works
without a tracking server running, which keeps the tests fast and means a
broken MLflow does not stop you from training locally.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .config import CONFIG, Config
from .train import TrainingRun

log = logging.getLogger(__name__)


def _client(cfg: Config):
    import mlflow

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    return mlflow


def champion_metrics(cfg: Config | None = None) -> dict[str, float]:
    """Metrics logged with whichever version currently serves production.

    Returns an empty dict when there is no champion, or when MLflow is not
    reachable. A missing registry should not be able to block a training run.
    """
    cfg = cfg or CONFIG
    try:
        _client(cfg)  # sets the tracking URI for the client below
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        versions = client.get_latest_versions(
            cfg.mlflow.registry_name, stages=[cfg.service.model_stage]
        )
        if not versions:
            return {}
        run = client.get_run(versions[0].run_id)
        return {k: float(v) for k, v in run.data.metrics.items()}
    except Exception as exc:
        log.warning("could not read champion metrics: %s", exc)
        return {}


def log_run(
    run: TrainingRun,
    cfg: Config | None = None,
    extra_params: dict[str, Any] | None = None,
) -> str | None:
    """Record the run, and register plus promote it only if the gate said so.

    Returns the MLflow run id, or None when tracking is unavailable. The caller
    treats None as "training succeeded, tracking did not", which is a warning
    rather than a failure.
    """
    cfg = cfg or CONFIG
    try:
        mlflow = _client(cfg)
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

            signature = _signature(run)
            mlflow.sklearn.log_model(
                run.pipeline,
                artifact_path="model",
                signature=signature,
                registered_model_name=cfg.mlflow.registry_name if run.decision.promote else None,
            )

            if run.decision.promote:
                _promote_latest(cfg)

            return active.info.run_id
    except Exception as exc:
        log.warning("MLflow logging skipped: %s", exc)
        return None


def _signature(run: TrainingRun):
    """Pin the input schema to the model so a wrong-shaped request fails at the door."""
    from mlflow.models.signature import infer_signature

    sample = run.reference.head(100)
    return infer_signature(sample, run.pipeline.predict(sample))


def _promote_latest(cfg: Config) -> None:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    versions = client.get_latest_versions(cfg.mlflow.registry_name, stages=["None"])
    if not versions:
        return
    newest = max(versions, key=lambda v: int(v.version))
    client.transition_model_version_stage(
        name=cfg.mlflow.registry_name,
        version=newest.version,
        stage=cfg.service.model_stage,
        archive_existing_versions=True,
    )
    log.info(
        "promoted %s version %s to %s",
        cfg.mlflow.registry_name,
        newest.version,
        cfg.service.model_stage,
    )


def load_champion(cfg: Config | None = None):
    """Load the serving model. Raises if there is none, because a service with
    no model should fail its readiness check rather than answer with nonsense."""
    cfg = cfg or CONFIG
    mlflow = _client(cfg)
    uri = f"models:/{cfg.mlflow.registry_name}/{cfg.service.model_stage}"
    return mlflow.sklearn.load_model(uri)


def load_reference(cfg: Config | None = None) -> pd.DataFrame | None:
    """The training window the champion saw, for the drift monitor to compare against."""
    cfg = cfg or CONFIG
    from .data import period_frames
    from .features import feature_frame

    try:
        reference = period_frames(cfg.data).reference
        return feature_frame(reference, cfg)
    except Exception as exc:
        log.warning("reference window unavailable: %s", exc)
        return None
