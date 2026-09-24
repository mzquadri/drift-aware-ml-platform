"""Every tunable the platform has, in one place.

Nothing reads an environment variable anywhere else. That keeps the training run,
the service and the Airflow tasks looking at the same numbers, which is the whole
reason training-serving skew is avoidable here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class DataConfig:
    """Where the raw data comes from and how it is cut into periods.

    The UCI Bike Sharing dataset covers 2011 and 2012. Training on 2011 and
    serving 2012 is not an artificial split: ridership grew substantially
    between the two years, so the second year really is drifted relative to the
    first. That is what makes the drift monitor worth having rather than a prop.
    """

    url: str = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
    # The first download records the archive's SHA-256 next to the extract.
    # Every later run re-checks it, so if upstream quietly republishes a
    # different file the run fails instead of training on changed data.
    member: str = "hour.csv"
    cache_dir: Path = ROOT / "data" / "raw"

    reference_year: int = 0  # dteday year flag in the dataset: 0 = 2011
    current_year: int = 1  # 1 = 2012

    target: str = "cnt"
    # Dropped on purpose: casual + registered sum exactly to cnt, so keeping
    # them would leak the label. instant is a row counter. dteday is redundant
    # once yr/mnth/weekday are present.
    leaky_columns: tuple[str, ...] = ("casual", "registered", "instant", "dteday")

    numeric_features: tuple[str, ...] = ("temp", "atemp", "hum", "windspeed")
    categorical_features: tuple[str, ...] = (
        "season",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
    )

    valid_fraction: float = 0.2
    random_seed: int = 17


@dataclass(frozen=True)
class TrainConfig:
    """Model and the gate a challenger has to clear to become champion."""

    n_estimators: int = _env_int("TRAIN_N_ESTIMATORS", 300)
    max_depth: int = _env_int("TRAIN_MAX_DEPTH", 14)
    min_samples_leaf: int = _env_int("TRAIN_MIN_SAMPLES_LEAF", 2)
    n_jobs: int = -1

    # A challenger is promoted only if it beats the champion's validation MAE by
    # at least this much. Without a margin you promote noise every other run.
    promotion_margin: float = _env_float("PROMOTION_MARGIN", 0.02)
    # And it must beat the naive "predict the training mean" baseline by this
    # much, so a degenerate dataset cannot quietly ship a useless model.
    baseline_margin: float = _env_float("BASELINE_MARGIN", 0.35)


@dataclass(frozen=True)
class DriftConfig:
    """When the monitor is allowed to say the world has moved."""

    # Share of monitored columns that must drift before we call it.
    dataset_drift_share: float = _env_float("DRIFT_DATASET_SHARE", 0.3)
    # Per-column p-value under the statistical test Evidently picks for the
    # column type. Lower means we demand stronger evidence.
    column_p_value: float = _env_float("DRIFT_COLUMN_P_VALUE", 0.05)
    # Never retrain on fewer rows than this; small windows drift by accident.
    min_window_rows: int = _env_int("DRIFT_MIN_WINDOW_ROWS", 500)


@dataclass(frozen=True)
class ServiceConfig:
    host: str = _env("SERVICE_HOST", "0.0.0.0")
    port: int = _env_int("SERVICE_PORT", 8000)
    model_name: str = _env("MODEL_NAME", "bike-demand")
    model_stage: str = _env("MODEL_STAGE", "Production")
    # Predictions are appended here so the drift job has real serving traffic to
    # compare against the training reference.
    prediction_log: Path = Path(_env("PREDICTION_LOG", str(ROOT / "data" / "predictions.parquet")))


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str = _env("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment: str = _env("MLFLOW_EXPERIMENT", "bike-demand")
    registry_name: str = _env("MODEL_NAME", "bike-demand")


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    @property
    def feature_columns(self) -> list[str]:
        return list(self.data.numeric_features) + list(self.data.categorical_features)


CONFIG = Config()
