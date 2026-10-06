"""Deciding whether the serving window has moved away from the training window.

Two things are deliberately separated: measuring drift, and acting on it.
Evidently produces the statistics; `decide_retrain` applies the policy. Keeping
them apart means the policy is testable without running a report, and the
thresholds live in config rather than being buried in a callback.

The policy watches the target as well as the features, because on this data the
features are not where the movement is. See `DriftConfig` for the numbers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CONFIG, Config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ColumnDrift:
    column: str
    method: str
    score: float
    threshold: float

    @property
    def is_p_value(self) -> bool:
        """Does a smaller number mean more drift?

        Evidently picks the test by how much reference data there is: above about a
        thousand rows it reports a distance, below that a p-value. The two run in
        opposite directions, and the method name is the only thing that says which
        arrived. It comes from the metric's `config`, so this reads a field rather
        than parsing a display name.
        """
        return "p_value" in self.method.lower()

    def exceeds(self, threshold: float) -> bool:
        """Is this score past `threshold`, in whichever direction the test runs?

        One rule in one place. A distance is drift when it is large; a p-value is
        drift when it is small, because a small p-value is the evidence that the two
        samples did not come from the same distribution.
        """
        return self.score < threshold if self.is_p_value else self.score > threshold

    @property
    def drifted(self) -> bool:
        return self.exceeds(self.threshold)

    def as_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "method": self.method,
            "score": round(self.score, 4),
            "threshold": self.threshold,
            "drifted": self.drifted,
        }


@dataclass(frozen=True)
class DriftResult:
    """What the monitor measured, before any decision is taken about it."""

    columns: list[ColumnDrift] = field(default_factory=list)
    target_drift: ColumnDrift | None = None
    n_reference: int = 0
    n_current: int = 0

    @property
    def drifted_columns(self) -> list[str]:
        return [c.column for c in self.columns if c.drifted]

    @property
    def feature_drift_share(self) -> float:
        if not self.columns:
            return 0.0
        return len(self.drifted_columns) / len(self.columns)

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_drift_share": round(self.feature_drift_share, 4),
            "drifted_columns": self.drifted_columns,
            "columns": [c.as_dict() for c in self.columns],
            "target_drift": self.target_drift.as_dict() if self.target_drift else None,
            "n_reference": self.n_reference,
            "n_current": self.n_current,
        }


@dataclass(frozen=True)
class RetrainDecision:
    retrain: bool
    reason: str
    drift: DriftResult

    def as_dict(self) -> dict[str, Any]:
        return {"retrain": self.retrain, "reason": self.reason, "drift": self.drift.as_dict()}


#: The column that says which season a row belongs to. Dropping to a bare row count
#: cannot express "comparable", because rows are not interchangeable across a year.
SEASON_COLUMN = "mnth"


def align_reference(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """Cut the reference down to the seasons the serving window actually covers.

    Comparing three weeks of January against a whole year reports that temperature,
    season, month and holiday have all moved, which is true and is not drift: it is
    the calendar. On this data that is 55-73% of features at every partial window,
    against the 9% the full-year comparison reports. The feature share only falls
    when the serving window has come round to the same seasons as the reference.

    The correction is to compare like with like - January against January, a year
    apart - which is the year-over-year alignment used for the same confound in
    seasonal churn models (arXiv:2608.18174). The cost there is needing a full prior
    year of history, and this project already has one: the reference *is* the prior
    year.

    A full-year serving window selects every month and the reference is returned
    whole, so the published figures are untouched. Only partial windows change.

    Alignment is skipped when there is no season column to align on, because
    returning an empty reference would be worse than comparing unlike windows.
    """
    del cfg  # kept for symmetry with the rest of the module
    if SEASON_COLUMN not in reference.columns or SEASON_COLUMN not in current.columns:
        return reference
    seasons = current[SEASON_COLUMN].unique()
    return reference[reference[SEASON_COLUMN].isin(seasons)]


def _definition(cfg: Config, extra_numeric: list[str] | None = None):
    """Tell Evidently which integer columns are really categories.

    season, month and hour are stored as integers but they are categories. Left
    to infer, Evidently would run a numeric distance over hour-of-day, which is
    close to meaningless.
    """
    from evidently import DataDefinition

    return DataDefinition(
        numerical_columns=list(cfg.data.numeric_features) + list(extra_numeric or []),
        categorical_columns=list(cfg.data.categorical_features),
    )


VALUE_DRIFT_TYPE = "evidently:metric_v2:ValueDrift"


def _parse(snapshot, target: str | None) -> tuple[list[ColumnDrift], ColumnDrift | None]:
    """Pull the per-column scores out of the snapshot.

    Read from each metric's `config` mapping, not from its display name. The
    name reorders its own fields depending on which arguments were passed, so
    splitting on it works until the day you pass one more argument and then
    silently does not.

    The method and threshold are read back rather than assumed, so what is
    reported is what Evidently actually applied.
    """
    columns: list[ColumnDrift] = []
    target_drift: ColumnDrift | None = None

    for metric in snapshot.dict().get("metrics", []):
        config = metric.get("config") or {}
        if config.get("type") != VALUE_DRIFT_TYPE:
            continue

        column = config.get("column")
        if column is None:
            continue

        entry = ColumnDrift(
            column=str(column),
            method=str(config.get("method", "unknown")),
            score=float(metric["value"]),
            threshold=float(config.get("threshold", 0.0)),
        )
        if target is not None and column == target:
            target_drift = entry
        else:
            columns.append(entry)

    return columns, target_drift


def _run_report(reference: pd.DataFrame, current: pd.DataFrame, cfg: Config):
    from evidently import Dataset, Report
    from evidently.presets import DataDriftPreset

    target = cfg.data.target
    has_target = target in reference.columns and target in current.columns

    columns = [c for c in cfg.feature_columns if c in reference.columns and c in current.columns]
    if has_target:
        columns = [*columns, target]

    definition = _definition(cfg, extra_numeric=[target] if has_target else None)
    report = Report(
        metrics=[
            DataDriftPreset(
                drift_share=cfg.drift.feature_drift_share,
                threshold=cfg.drift.column_threshold,
            )
        ]
    )
    snapshot = report.run(
        current_data=Dataset.from_pandas(current[columns], data_definition=definition),
        reference_data=Dataset.from_pandas(reference[columns], data_definition=definition),
    )
    return snapshot, (target if has_target else None)


def measure(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    cfg: Config | None = None,
) -> DriftResult:
    """Score every shared column, and the target too when both frames carry it.

    Evidently picks the test per column type: a normed Wasserstein distance for
    numeric columns, Jensen-Shannon for categorical ones. Both are distances, so
    a larger score means more drift.
    """
    cfg = cfg or CONFIG
    snapshot, target = _run_report(reference, current, cfg)
    column_drifts, target_drift = _parse(snapshot, target)
    return DriftResult(
        columns=column_drifts,
        target_drift=target_drift,
        n_reference=len(reference),
        n_current=len(current),
    )


def decide_retrain(result: DriftResult, cfg: Config | None = None) -> RetrainDecision:
    """Policy, kept separate from measurement so it can be tested on its own.

    Order matters. The window guard runs first, because a handful of rows fails
    a distribution test for reasons that have nothing to do with the world
    moving, and retraining on that makes the model worse. The target check runs
    before the feature check, because on this data it is the signal that fires.
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

    # The same guard on the other side. Once the reference is cut to the seasons being
    # served it can be thin, and a comparison against a handful of comparable rows
    # fails a distribution test for the same reason a tiny serving window does.
    if result.n_reference < cfg.drift.min_reference_rows:
        return RetrainDecision(
            retrain=False,
            reason=(
                f"only {result.n_reference} rows of reference cover the seasons being "
                f"served, below the {cfg.drift.min_reference_rows} needed before the "
                "comparison stops rejecting differences too small to act on"
            ),
            drift=result,
        )

    target = result.target_drift
    if target is not None and target.exceeds(cfg.drift.target_threshold):
        # Say which way the comparison ran. "0.000, over the threshold" is false for a
        # p-value, and the reason line is what someone reads at three in the morning.
        side = "under" if target.is_p_value else "over"
        score = f"{target.score:.3g}" if target.is_p_value else f"{target.score:.3f}"
        return RetrainDecision(
            retrain=True,
            reason=(
                f"target '{target.column}' drifted {score} by {target.method}, "
                f"{side} the {cfg.drift.target_threshold} threshold, while only "
                f"{result.feature_drift_share:.0%} of features moved; the relationship the "
                f"model learned no longer holds"
            ),
            drift=result,
        )

    share = result.feature_drift_share
    if share >= cfg.drift.feature_drift_share:
        return RetrainDecision(
            retrain=True,
            reason=(
                f"{share:.0%} of features drifted ({', '.join(result.drifted_columns)}), "
                f"at or over the {cfg.drift.feature_drift_share:.0%} threshold"
            ),
            drift=result,
        )

    detail = f"the target moved {target.score:.3f}" if target else "the target was not observed"
    return RetrainDecision(
        retrain=False,
        reason=(
            f"{share:.0%} of features drifted, under the "
            f"{cfg.drift.feature_drift_share:.0%} threshold, and {detail}"
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

    When a retrain fires at three in the morning the thing you want is the
    report showing which columns moved, not a log line saying "drift".
    """
    cfg = cfg or CONFIG
    snapshot, _ = _run_report(reference, current, cfg)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))
    return html_path


def write_decision(decision: RetrainDecision, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision.as_dict(), indent=2) + "\n", encoding="utf-8")
    return path
