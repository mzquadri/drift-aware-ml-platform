"""Feature construction, with exactly one function allowed to fit.

Training-serving skew usually starts with a second place that fits a transformer.
Here `build_pipeline` is the only thing that ever calls fit, and the fitted
pipeline travels with the model, so serving cannot drift away from training by
accident.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CONFIG, Config


class SchemaError(ValueError):
    """Raised when a frame does not carry the columns the pipeline needs."""


def require_columns(frame: pd.DataFrame, cfg: Config | None = None) -> None:
    cfg = cfg or CONFIG
    missing = [c for c in cfg.feature_columns if c not in frame.columns]
    if missing:
        raise SchemaError(f"missing feature columns: {missing}")


def feature_frame(frame: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or CONFIG
    require_columns(frame, cfg)
    return frame[cfg.feature_columns].copy()


def build_pipeline(cfg: Config | None = None) -> Pipeline:
    """Preprocessing and the estimator as one object.

    One-hot for the calendar columns because hour-of-day and month are cyclical
    categories, not magnitudes; treating hour 23 as "larger than" hour 1 is the
    kind of thing that quietly costs accuracy.
    """
    cfg = cfg or CONFIG
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(cfg.data.numeric_features)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(cfg.data.categorical_features),
            ),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=cfg.train.n_estimators,
        max_depth=cfg.train.max_depth,
        min_samples_leaf=cfg.train.min_samples_leaf,
        n_jobs=cfg.train.n_jobs,
        random_state=cfg.data.random_seed,
    )
    return Pipeline([("features", pre), ("model", model)])


def fit_pipeline(train: pd.DataFrame, cfg: Config | None = None) -> tuple[Pipeline, pd.DataFrame]:
    """The only place in the codebase that fits anything.

    Returns the fitted pipeline and the exact frame it saw, which becomes the
    drift monitor's reference window.
    """
    cfg = cfg or CONFIG
    x = feature_frame(train, cfg)
    y = train[cfg.data.target]
    pipeline = build_pipeline(cfg)
    pipeline.fit(x, y)
    return pipeline, x
