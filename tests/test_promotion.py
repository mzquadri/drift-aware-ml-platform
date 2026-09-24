"""The promotion gate is the only thing standing between a bad model and production,
so it gets tested on its own, with no data, no registry and no network."""

from __future__ import annotations

import pandas as pd
import pytest

from mlplatform.config import CONFIG
from mlplatform.train import Scores, baseline_mae, decide


def scores(mae: float) -> Scores:
    return Scores(mae=mae, rmse=mae * 1.3, r2=0.8, n_rows=1000)


def test_model_that_cannot_beat_the_mean_is_refused():
    decision = decide(scores(mae=95.0), base_mae=100.0, champion_mae=None)
    assert decision.promote is False
    assert "baseline" in decision.reason


def test_first_model_is_promoted_once_it_clears_the_baseline():
    # baseline_margin defaults to 0.35, so it must be at least 35% better
    decision = decide(scores(mae=50.0), base_mae=100.0, champion_mae=None)
    assert decision.promote is True
    assert decision.champion_mae is None


def test_challenger_only_marginally_better_is_refused():
    """A 1% gain is noise. Promoting it churns production for nothing."""
    decision = decide(scores(mae=49.5), base_mae=100.0, champion_mae=50.0)
    assert decision.promote is False
    assert "not" in decision.reason


def test_challenger_clearly_better_is_promoted():
    decision = decide(scores(mae=45.0), base_mae=100.0, champion_mae=50.0)
    assert decision.promote is True


def test_worse_challenger_never_wins():
    decision = decide(scores(mae=60.0), base_mae=100.0, champion_mae=50.0)
    assert decision.promote is False


@pytest.mark.parametrize("champion", [None, 50.0])
def test_decision_round_trips_to_a_dict(champion):
    payload = decide(scores(mae=40.0), base_mae=100.0, champion_mae=champion).as_dict()
    assert set(payload) == {"promote", "reason", "challenger", "baseline_mae", "champion_mae"}
    assert payload["challenger"]["mae"] == 40.0


def test_baseline_is_the_training_mean_not_the_validation_mean():
    """Using the validation mean would leak the answer into the baseline and make
    the gate easier to pass than it should be."""
    train = pd.DataFrame({CONFIG.data.target: [10.0] * 100})
    valid = pd.DataFrame({CONFIG.data.target: [30.0] * 100})
    assert baseline_mae(train, valid, CONFIG) == pytest.approx(20.0)
