"""Deciding whether the serving window has moved away from the training window.

Two things are deliberately separated here: measuring drift, and acting on it.
Evidently produces the statistics; `decide_retrain` applies the policy. Keeping
them apart means the policy is testable without running a report, and the
thresholds live in config rather than being buried in a callback.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CONFIG, Config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DriftResult:
    """What the monitor measured, before any decision is taken about it."""

    dataset_drift: bool
    drifted_share: float
    drifted_columns: list[str]
    n_reference: int
    n_current: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_drift": self.dataset_drift,
            "drifted_share": self.drifted_share,
            "drifted_columns": self.drifted_columns,
            "n_reference": self.n_reference,
            "n_current": self.n_current,
        }


@dataclass(frozen=True)
class RetrainDecision:
    retrain: bool
    reason: str
    drift: DriftResult


def measure(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    cfg: Config | None = None,
) -> DriftResult:
    """Run Evidently's data drift preset over the feature columns.

    Evidently picks a test per column type, a two sample Kolmogorov-Smirnov for
    continuous columns and a chi-squared for categorical ones, which is why the
    threshold below is expressed as a p-value rather than a distance.
    """
    cfg = cfg or CONFIG
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    columns = [c for c in cfg.feature_columns if c in reference.columns and c in current.columns]
    report = Report(metrics=[DataDriftPreset(stattest_threshold=cfg.drift.column_p_value)])
    report.run(reference_data=reference[columns], current_data=current[columns])

    payload = report.as_dict()
    result = payload["metrics"][0]["result"]
    by_column = result.get("drift_by_columns", {})
    drifted = sorted(name for name, info in by_column.items() if info.get("drift_detected"))
    share = float(result.get("share_of_drifted_columns", 0.0))

    return DriftResult(
        dataset_drift=share >= cfg.drift.dataset_drift_share,
        drifted_share=share,
        drifted_columns=drifted,
        n_reference=len(reference),
        n_current=len(current),
    )


def decide_retrain(result: DriftResult, cfg: Config | None = None) -> RetrainDecision:
    """Policy, kept separate from measurement so it can be tested on its own.

    The window size guard matters more than it looks. A handful of rows will
    fail a distribution test for reasons that have nothing to do with the world
    changing, and retraining on them makes the model worse, not better.
    """
    cfg = cfg or CONFIG

    if result.n_current < cfg.drift.min_window_rows:
        return RetrainDecision(
            retrain=False,
            reason=(
                f"serving window has {result.n_current} rows, below the "
                f"{cfg.drift.min_window_rows} needed for the test to mean anything"
            ),
            drift=result,
        )

    if not result.dataset_drift:
        return RetrainDecision(
            retrain=False,
            reason=(
                f"{result.drifted_share:.0%} of columns drifted, under the "
                f"{cfg.drift.dataset_drift_share:.0%} threshold"
            ),
            drift=result,
        )

    return RetrainDecision(
        retrain=True,
        reason=(
            f"{result.drifted_share:.0%} of columns drifted ({', '.join(result.drifted_columns)}), "
            f"at or over the {cfg.drift.dataset_drift_share:.0%} threshold"
        ),
        drift=result,
    )


def write_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    html_path: Path,
    cfg: Config | None = None,
) -> Path:
    """Save the human-readable Evidently report next to the decision.

    When a retrain fires at three in the morning, the thing you want is the
    report that explains which columns moved, not a log line saying "drift".
    """
    cfg = cfg or CONFIG
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.report import Report

    columns = [c for c in cfg.feature_columns if c in reference.columns and c in current.columns]
    metrics = [DataDriftPreset(stattest_threshold=cfg.drift.column_p_value)]
    if cfg.data.target in reference.columns and cfg.data.target in current.columns:
        columns = [*columns, cfg.data.target]
        metrics.append(TargetDriftPreset())

    report = Report(metrics=metrics)
    report.run(reference_data=reference[columns], current_data=current[columns])
    html_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(html_path))
    return html_path


def write_decision(decision: RetrainDecision, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "retrain": decision.retrain,
        "reason": decision.reason,
        "drift": decision.drift.as_dict(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
