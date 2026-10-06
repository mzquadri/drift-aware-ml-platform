"""The detector sweep, tested without a detector library or a dataset.

Everything here takes a factory, so the sweep logic is exercised with a fake whose
behaviour is obvious. The point of the experiment is a comparison between detectors, and
a comparison is only worth reading if the harness that produced it is not itself the
variable.
"""

from __future__ import annotations

from mlplatform.experiments.latency import Alarms, calibrate, scan


class Threshold:
    """Fires whenever a value crosses a bound. Deterministic, no library needed."""

    def __init__(self, bound: float) -> None:
        self._bound = bound
        self.drift_detected = False

    def update(self, value: float) -> None:
        self.drift_detected = value > self._bound


def test_a_quiet_stream_produces_no_alarms() -> None:
    result = scan(lambda: Threshold(10.0), [1.0] * 50)

    assert result == Alarms(first=None, count=0, length=50)


def test_the_first_alarm_is_the_index_that_crossed() -> None:
    stream = [1.0] * 20 + [99.0] + [1.0] * 10

    result = scan(lambda: Threshold(10.0), stream)

    assert result.first == 20
    assert result.count == 1


def test_every_crossing_is_counted_not_just_the_first() -> None:
    result = scan(lambda: Threshold(10.0), [1.0, 99.0, 1.0, 99.0, 99.0])

    assert result.count == 3


def test_each_scan_starts_from_a_fresh_detector() -> None:
    # A detector carried between streams would make the control's state leak into the
    # treatment, and the comparison would measure the order they were run in.
    factory_calls = []

    def factory() -> Threshold:
        factory_calls.append(1)
        return Threshold(10.0)

    scan(factory, [1.0])
    scan(factory, [1.0])

    assert len(factory_calls) == 2


def test_calibration_picks_the_most_sensitive_setting_that_stays_silent() -> None:
    # Calibrate on in-regime data only. Choosing by what it does to the drifted stream
    # would be fitting the detector to the answer.
    control = [1.0] * 20 + [5.0]

    chosen = calibrate(
        lambda bound: lambda: Threshold(bound),
        control=control,
        candidates=[1.0, 3.0, 4.0, 10.0, 50.0],
    )

    # 4.0 still fires on the 5.0, so the first silent candidate is 10.0.
    assert chosen == 10.0


def test_calibration_returns_none_when_nothing_stays_silent() -> None:
    chosen = calibrate(
        lambda bound: lambda: Threshold(bound),
        control=[100.0] * 10,
        candidates=[1.0, 2.0],
    )

    assert chosen is None
