"""Drift policy, tested without Evidently.

`measure` needs the library and real frames; `decide_retrain` is pure policy and
is where the mistakes that cost you money actually live, so it is tested alone.
"""

from __future__ import annotations

from mlplatform.config import CONFIG
from mlplatform.drift import DriftResult, decide_retrain


def result(share: float, n_current: int = 5000, columns: list[str] | None = None) -> DriftResult:
    return DriftResult(
        dataset_drift=share >= CONFIG.drift.dataset_drift_share,
        drifted_share=share,
        drifted_columns=columns or ["temp", "hum"],
        n_reference=8000,
        n_current=n_current,
    )


def test_a_tiny_window_never_triggers_retraining():
    """Forty rows will fail a distribution test for reasons that are not drift."""
    decision = decide_retrain(result(share=0.9, n_current=40))
    assert decision.retrain is False
    assert "rows" in decision.reason


def test_quiet_period_does_not_retrain():
    decision = decide_retrain(result(share=0.05))
    assert decision.retrain is False
    assert "threshold" in decision.reason


def test_real_drift_on_a_large_window_retrains():
    decision = decide_retrain(result(share=0.6))
    assert decision.retrain is True
    assert "temp" in decision.reason


def test_threshold_is_inclusive():
    """Exactly at the threshold counts as drifted; otherwise the configured number
    means something slightly different from what it says."""
    decision = decide_retrain(result(share=CONFIG.drift.dataset_drift_share))
    assert decision.retrain is True


def test_window_guard_beats_a_strong_drift_signal():
    """Order matters: the size check runs before the drift check, so a small
    window cannot retrain no matter how dramatic the measured shift."""
    decision = decide_retrain(result(share=1.0, n_current=CONFIG.drift.min_window_rows - 1))
    assert decision.retrain is False
    assert "rows" in decision.reason
