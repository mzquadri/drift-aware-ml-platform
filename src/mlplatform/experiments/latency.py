"""Running a drift detector over a stream and recording when it first speaks.

Kept separate from `drift.py` for the same reason measurement is kept separate from
policy there: this is an experiment about detectors, and the harness that runs it should
not be the variable under test. Detectors arrive as factories so a fresh one is built per
stream; a detector carried from the control into the treatment would make the result
depend on the order they were run in.

Nothing here imports a detector library. The experiment runner supplies river's; the
tests supply a two-line fake.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class Detector(Protocol):
    """The shape river's detectors already have."""

    drift_detected: bool

    def update(self, value: float) -> Any: ...


Factory = Callable[[], Detector]


@dataclass(frozen=True)
class Alarms:
    """What a detector did over one stream."""

    #: Index of the first alarm, or None if it never fired.
    first: int | None
    count: int
    length: int

    def as_dict(self) -> dict[str, Any]:
        return {"first": self.first, "count": self.count, "length": self.length}


def scan(factory: Factory, stream: Iterable[float]) -> Alarms:
    """Run one fresh detector over one stream."""
    detector = factory()
    first: int | None = None
    count = 0
    length = 0

    for index, value in enumerate(stream):
        length = index + 1
        detector.update(float(value))
        if detector.drift_detected:
            count += 1
            if first is None:
                first = index

    return Alarms(first=first, count=count, length=length)


def calibrate(
    build: Callable[[float], Factory],
    control: Sequence[float],
    candidates: Sequence[float],
) -> float | None:
    """Pick the most sensitive setting that raises no alarm on in-regime data.

    Calibration uses the control only. Choosing by what a setting does to the drifted
    stream would be selecting the detector against the answer, which is how any detector
    is made to look good on any dataset.

    `candidates` are ordered most sensitive first and the first silent one wins: a more
    sensitive setting detects sooner, and there is no reason to buy more caution than the
    control demands. Which direction counts as sensitive depends on the detector, so the
    ordering is the caller's to get right.
    """
    for candidate in candidates:
        if scan(build(candidate), control).count == 0:
            return candidate
    return None


__all__ = ["Alarms", "Detector", "Factory", "calibrate", "scan"]
