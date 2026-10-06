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


# ----------------------------------------------- which way the comparison runs

P_VALUE = "K-S p_value"
DISTANCE = "Wasserstein distance (normed)"


def scored(method: str, score: float, threshold: float = THRESHOLD) -> ColumnDrift:
    return ColumnDrift(column="cnt", method=method, score=score, threshold=threshold)


def test_a_distance_is_drift_when_it_is_large():
    assert scored(DISTANCE, 0.9).drifted is True
    assert scored(DISTANCE, 0.001).drifted is False


def test_a_p_value_is_drift_when_it_is_small():
    """Evidently returns a p-value once the reference falls to about a thousand rows.

    A p-value runs the other way: small means the distributions differ. Comparing it
    the same way as a distance inverts the verdict in both directions, so identical
    data reads as drifted and overwhelming drift reads as quiet.
    """
    assert scored(P_VALUE, 1.6e-23).drifted is True
    assert scored(P_VALUE, 1.0).drifted is False


def test_the_policy_reads_a_p_value_target_the_right_way_round():
    target = scored(P_VALUE, 1.6e-23)
    quiet = DriftResult(
        columns=[col("f0", 0.01)], target_drift=target, n_reference=8000, n_current=5000
    )

    decision = decide_retrain(quiet)

    assert decision.retrain is True
    assert "cnt" in decision.reason


def test_the_policy_leaves_a_quiet_p_value_alone():
    target = scored(P_VALUE, 0.94)
    quiet = DriftResult(
        columns=[col("f0", 0.01)], target_drift=target, n_reference=8000, n_current=5000
    )

    assert decide_retrain(quiet).retrain is False


def test_a_distance_target_still_behaves_as_it_did():
    # The default configuration uses a full year of reference and therefore a distance.
    # Nothing about that path may move.
    loud = DriftResult(
        columns=[col("f0", 0.01)], target_drift=col("cnt", 0.679), n_reference=8000, n_current=5000
    )
    hushed = DriftResult(
        columns=[col("f0", 0.01)], target_drift=col("cnt", 0.02), n_reference=8000, n_current=5000
    )

    assert decide_retrain(loud).retrain is True
    assert decide_retrain(hushed).retrain is False


def test_the_reason_describes_the_comparison_that_was_actually_made():
    """The reason is what someone reads at three in the morning; it has to be true.

    "0.000, over the 0.2 threshold" is false for a p-value, which is drift precisely
    because it fell below.
    """
    p = DriftResult(
        columns=[col("f0", 0.01)],
        target_drift=scored(P_VALUE, 1.6e-23),
        n_reference=8000,
        n_current=5000,
    )
    d = DriftResult(
        columns=[col("f0", 0.01)], target_drift=col("cnt", 0.679), n_reference=8000, n_current=5000
    )

    assert "under" in decide_retrain(p).reason
    assert "over" not in decide_retrain(p).reason
    assert "over" in decide_retrain(d).reason


# ----------------------------------------------- comparing like seasons with like

import pandas as pd  # noqa: E402

from mlplatform.drift import align_reference  # noqa: E402


def frame(months: list[int], rows_per_month: int = 10) -> pd.DataFrame:
    rows = [{"mnth": m, "cnt": m * 10 + i} for m in months for i in range(rows_per_month)]
    return pd.DataFrame(rows)


def test_the_reference_is_cut_down_to_the_months_being_served():
    reference = frame(list(range(1, 13)))
    current = frame([1])

    aligned = align_reference(reference, current)

    assert sorted(aligned["mnth"].unique()) == [1]
    assert len(aligned) == 10


def test_a_full_year_window_leaves_the_reference_alone():
    """The published figures are full-year, and alignment must not disturb them."""
    reference = frame(list(range(1, 13)))
    current = frame(list(range(1, 13)))

    aligned = align_reference(reference, current)

    assert len(aligned) == len(reference)


def test_alignment_is_skipped_when_the_season_column_is_absent():
    # Not every frame carries mnth. Dropping every row would be worse than not aligning.
    reference = pd.DataFrame({"cnt": [1, 2, 3]})
    current = pd.DataFrame({"cnt": [4, 5]})

    assert len(align_reference(reference, current)) == 3


def test_a_month_the_reference_never_saw_leaves_nothing_to_compare():
    # Serving a season absent from the reference is not drift, it is no evidence.
    reference = frame([1, 2])
    current = frame([7])

    assert len(align_reference(reference, current)) == 0


def test_the_policy_refuses_when_the_comparable_reference_is_too_thin():
    thin = DriftResult(
        columns=[col("f0", 0.9)],
        target_drift=col("cnt", 0.9),
        n_reference=40,
        n_current=5000,
    )

    decision = decide_retrain(thin)

    assert decision.retrain is False
    assert "reference" in decision.reason


def test_a_reference_too_small_for_a_distance_test_is_refused():
    """688 rows of January is comparable but not enough to judge with.

    Below about a thousand rows Evidently answers with a p-value, and a p-value at
    these sample sizes rejects differences too small to act on: aligning January
    against January leaves 688 reference rows and 64% of features "drifting". The
    honest answer there is that there is not enough comparable history yet.
    """
    thin = DriftResult(
        columns=[col("f0", 0.9)],
        target_drift=col("cnt", 0.9),
        n_reference=688,
        n_current=500,
    )

    decision = decide_retrain(thin)

    assert decision.retrain is False
    assert "reference" in decision.reason


def test_enough_comparable_reference_lets_the_decision_through():
    ok = DriftResult(
        columns=[col("f0", 0.01)],
        target_drift=col("cnt", 0.9),
        n_reference=2067,
        n_current=1500,
    )

    assert decide_retrain(ok).retrain is True
