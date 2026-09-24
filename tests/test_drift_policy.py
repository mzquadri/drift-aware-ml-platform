"""Drift policy, tested without Evidently.

`measure` needs the library and real frames; `decide_retrain` is pure policy and
is where the expensive mistakes live, so it is tested on its own.

The case that matters most on this dataset: the features stay put while the
target moves. A policy that only watched features would sleep through it.
"""

from __future__ import annotations

from mlplatform.config import CONFIG
from mlplatform.drift import ColumnDrift, DriftResult, decide_retrain

THRESHOLD = CONFIG.drift.column_threshold


def col(name: str, score: float) -> ColumnDrift:
    return ColumnDrift(
        column=name, method="Wasserstein distance (normed)", score=score, threshold=THRESHOLD
    )


def result(
    scores: list[float],
    target_score: float | None = None,
    n_current: int = 5000,
) -> DriftResult:
    columns = [col(f"f{i}", s) for i, s in enumerate(scores)]
    target = col(CONFIG.data.target, target_score) if target_score is not None else None
    return DriftResult(columns=columns, target_drift=target, n_reference=8000, n_current=n_current)


def test_a_tiny_window_never_triggers_retraining():
    """Forty rows fail a distribution test for reasons that are not drift."""
    decision = decide_retrain(result([0.9, 0.9, 0.9], target_score=0.9, n_current=40))
    assert decision.retrain is False
    assert "rows" in decision.reason


def test_quiet_period_does_not_retrain():
    decision = decide_retrain(result([0.01, 0.02, 0.01], target_score=0.02))
    assert decision.retrain is False
    assert "under the" in decision.reason


def test_target_drift_alone_is_enough():
    """The real case on this data: humidity aside, the covariates hold still while
    ridership rises 63%. Watching features only would miss it entirely."""
    decision = decide_retrain(result([0.01, 0.17, 0.02, 0.03], target_score=0.68))
    assert decision.retrain is True
    assert "target" in decision.reason
    # and it should say so honestly: most features did not move
    assert "25% of features" in decision.reason


def test_feature_drift_alone_is_enough():
    """With no target available, broad feature movement still retrains."""
    decision = decide_retrain(result([0.5, 0.6, 0.7, 0.01]))
    assert decision.retrain is True
    assert "features drifted" in decision.reason


def test_feature_threshold_is_inclusive():
    """Exactly at the threshold counts, otherwise the configured number means
    something slightly different from what it says."""
    # 3 of 10 drifted == 0.30, the default share
    scores = [0.5, 0.5, 0.5] + [0.0] * 7
    decision = decide_retrain(result(scores))
    assert decision.retrain is True


def test_window_guard_beats_every_other_signal():
    """Order matters: the size check runs first, so a small window cannot
    retrain however dramatic the measured shift."""
    decision = decide_retrain(
        result([1.0, 1.0], target_score=1.0, n_current=CONFIG.drift.min_window_rows - 1)
    )
    assert decision.retrain is False
    assert "rows" in decision.reason


def test_decision_serialises_for_the_airflow_branch():
    """The DAG branches on this file, so its shape is part of the contract."""
    payload = decide_retrain(result([0.01], target_score=0.68)).as_dict()
    assert payload["retrain"] is True
    assert payload["drift"]["target_drift"]["column"] == CONFIG.data.target
    assert payload["drift"]["target_drift"]["drifted"] is True
    assert "feature_drift_share" in payload["drift"]


def test_share_is_zero_when_no_columns_were_scored():
    """Guards a division by zero on an empty comparison."""
    assert DriftResult().feature_drift_share == 0.0
