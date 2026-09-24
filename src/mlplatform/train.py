"""Train a challenger, score it, and decide whether it deserves to be champion.

The decision is the point of this module. Training a model is easy; refusing to
ship one that is not actually better is the part that keeps a platform honest.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .config import CONFIG, Config
from .data import period_frames, train_valid_split
from .features import feature_frame, fit_pipeline

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Scores:
    mae: float
    rmse: float
    r2: float
    n_rows: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionDecision:
    """Why a model was or was not promoted, in a form you can log and read back."""

    promote: bool
    reason: str
    challenger: Scores
    baseline_mae: float
    champion_mae: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "promote": self.promote,
            "reason": self.reason,
            "challenger": self.challenger.as_dict(),
            "baseline_mae": self.baseline_mae,
            "champion_mae": self.champion_mae,
        }


def score(pipeline: Pipeline, frame: pd.DataFrame, cfg: Config | None = None) -> Scores:
    cfg = cfg or CONFIG
    x = feature_frame(frame, cfg)
    y_true = frame[cfg.data.target].to_numpy()
    y_pred = pipeline.predict(x)
    return Scores(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
        n_rows=len(frame),
    )


def baseline_mae(train: pd.DataFrame, valid: pd.DataFrame, cfg: Config | None = None) -> float:
    """What you get by predicting the training mean for every hour.

    Any model that cannot beat this by a clear margin has learned nothing, and
    the promotion gate below treats that as a failed run rather than a shipped
    model.
    """
    cfg = cfg or CONFIG
    guess = float(train[cfg.data.target].mean())
    truth = valid[cfg.data.target].to_numpy()
    return float(mean_absolute_error(truth, np.full_like(truth, guess, dtype=float)))


def decide(
    challenger: Scores,
    base_mae: float,
    champion_mae: float | None,
    cfg: Config | None = None,
) -> PromotionDecision:
    """Two gates, both of which have to pass.

    The baseline gate catches a broken dataset or a model that failed to learn.
    The margin gate stops a challenger that is only noise-better from churning
    production every night.
    """
    cfg = cfg or CONFIG

    required = base_mae * (1.0 - cfg.train.baseline_margin)
    if challenger.mae > required:
        return PromotionDecision(
            promote=False,
            reason=(
                f"challenger MAE {challenger.mae:.2f} does not beat the mean baseline "
                f"{base_mae:.2f} by the required {cfg.train.baseline_margin:.0%} "
                f"(needed <= {required:.2f})"
            ),
            challenger=challenger,
            baseline_mae=base_mae,
            champion_mae=champion_mae,
        )

    if champion_mae is None:
        return PromotionDecision(
            promote=True,
            reason="no champion registered yet, promoting the first model that clears the baseline",
            challenger=challenger,
            baseline_mae=base_mae,
            champion_mae=None,
        )

    needed = champion_mae * (1.0 - cfg.train.promotion_margin)
    if challenger.mae <= needed:
        return PromotionDecision(
            promote=True,
            reason=(
                f"challenger MAE {challenger.mae:.2f} beats champion {champion_mae:.2f} "
                f"by at least {cfg.train.promotion_margin:.0%}"
            ),
            challenger=challenger,
            baseline_mae=base_mae,
            champion_mae=champion_mae,
        )

    return PromotionDecision(
        promote=False,
        reason=(
            f"challenger MAE {challenger.mae:.2f} is not {cfg.train.promotion_margin:.0%} "
            f"better than champion {champion_mae:.2f} (needed <= {needed:.2f})"
        ),
        challenger=challenger,
        baseline_mae=base_mae,
        champion_mae=champion_mae,
    )


@dataclass
class TrainingRun:
    pipeline: Pipeline
    reference: pd.DataFrame
    valid_scores: Scores
    decision: PromotionDecision


def run_training(frame: pd.DataFrame | None = None, cfg: Config | None = None) -> TrainingRun:
    """Fit on the reference period and report whether the result is worth shipping.

    Passing a frame is how the retraining job feeds in a window that includes
    recent, drifted data instead of only the original reference year.
    """
    cfg = cfg or CONFIG
    if frame is None:
        frame = period_frames(cfg.data).reference

    train, valid = train_valid_split(frame, cfg.data)
    pipeline, reference = fit_pipeline(train, cfg)

    scores = score(pipeline, valid, cfg)
    base = baseline_mae(train, valid, cfg)
    champion = _champion_mae(cfg)
    decision = decide(scores, base, champion, cfg)

    log.info("validation %s", scores.as_dict())
    log.info("promotion: %s", decision.reason)
    return TrainingRun(
        pipeline=pipeline, reference=reference, valid_scores=scores, decision=decision
    )


def _champion_mae(cfg: Config) -> float | None:
    """Read the current champion's validation MAE from the registry, if there is one.

    Kept isolated so the training logic can be tested without a registry running.
    """
    try:
        from .registry import champion_metrics
    except ImportError:  # pragma: no cover - registry is optional locally
        return None
    metrics = champion_metrics(cfg)
    if not metrics:
        return None
    value = metrics.get("valid_mae")
    return float(value) if value is not None else None


def write_run_summary(run: TrainingRun, path: Path) -> Path:
    """One JSON file per run, so a failed promotion is auditable after the fact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "valid": run.valid_scores.as_dict(),
        "decision": run.decision.as_dict(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
