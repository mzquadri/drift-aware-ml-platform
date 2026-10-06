"""How long each way of noticing drift takes to notice it.

The platform watches distributions over a window: it compares the serving period against
the training period and applies a policy. The streaming literature does something else -
it watches the model's own error one observation at a time, with detectors like ADWIN,
Page-Hinkley and KSWIN. The second kind is the one that looks at P(y|X) rather than at a
marginal, so it is worth knowing whether it would have said anything sooner.

This runs both over the same 2012 and writes what happened to results/drift_latency.json.

Two things make the comparison honest rather than flattering.

A control. The same model is streamed over held-out 2011, which is the regime it was
trained on, so a detector that fires there is raising a false alarm rather than finding
anything. A detector evaluated only on the drifted stream always looks good.

Calibration on the control alone. ADWIN's sensitivity is chosen as the most sensitive
setting that stays silent on 2011, never by what it does to 2012. Picking the setting
that detects 2012 best would be choosing the detector against the answer.

    python scripts/experiments/drift_latency.py

Needs the dataset, so it reaches the network on a cold cache. The test suite does not.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mlplatform.config import CONFIG  # noqa: E402
from mlplatform.data import ensure_raw, period_frames  # noqa: E402
from mlplatform.drift import align_reference, decide_retrain, measure  # noqa: E402
from mlplatform.experiments.latency import calibrate, scan  # noqa: E402
from mlplatform.features import feature_frame, fit_pipeline  # noqa: E402

OUT = ROOT / "results" / "drift_latency.json"

#: Most sensitive first. ADWIN's delta is a confidence bound: larger reacts sooner.
ADWIN_DELTAS = (0.002, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10)

#: Where to probe the window monitor. Hourly data, so 24 rows is a day.
WINDOW_PROBES = (168, 336, 500, 720, 1000, 1500, 1855, 2500, 4000, 8734)

#: Where the aligned comparison first clears the reference guard and fires honestly.


def _digest_of_data() -> str:
    import hashlib

    return hashlib.sha256(ensure_raw().read_bytes()).hexdigest()


def main() -> int:
    logging.disable(logging.WARNING)
    periods = period_frames()
    reference, current = periods.reference, periods.current
    target = CONFIG.data.target

    # The model under test is the champion: trained on 2011 only. The control is the
    # slice of 2011 it never saw, which is the same split the promotion gate scores on.
    cut = int(len(reference) * (1 - CONFIG.data.valid_fraction))
    pipeline, _ = fit_pipeline(reference.iloc[:cut])

    held = reference.iloc[cut:]
    control = np.abs(held[target].to_numpy() - pipeline.predict(feature_frame(held)))
    treatment = np.abs(current[target].to_numpy() - pipeline.predict(feature_frame(current)))

    from river import drift

    detectors: dict[str, Any] = {
        "PageHinkley": lambda: drift.PageHinkley(),
        "ADWIN": lambda: drift.ADWIN(),
        "KSWIN": lambda: drift.KSWIN(seed=CONFIG.data.random_seed),
    }

    at_defaults = {
        name: {
            "control": scan(factory, control).as_dict(),
            "current": scan(factory, treatment).as_dict(),
        }
        for name, factory in detectors.items()
    }

    sweep = []
    for delta in ADWIN_DELTAS:
        sweep.append(
            {
                "delta": delta,
                "control": scan(lambda d=delta: drift.ADWIN(delta=d), control).as_dict(),
                "current": scan(lambda d=delta: drift.ADWIN(delta=d), treatment).as_dict(),
            }
        )

    chosen = calibrate(
        lambda d: lambda: drift.ADWIN(delta=d),
        control=list(control),
        candidates=list(ADWIN_DELTAS),
    )
    calibrated = scan(lambda: drift.ADWIN(delta=chosen), treatment) if chosen else None

    # The window monitor, probed at the same points, with the seasonal coverage that
    # explains what it does at small windows.
    window = []
    for rows in WINDOW_PROBES:
        if rows > len(current):
            continue
        serving = current.iloc[:rows]
        # Both comparisons, because the difference between them is the finding:
        # unaligned reports the calendar, aligned reports the world.
        unaligned = measure(reference, serving)
        comparable = align_reference(reference, serving)
        aligned = measure(comparable, serving)
        decision = decide_retrain(aligned)
        window.append(
            {
                "rows": rows,
                "days": round(rows / 24, 1),
                "months_present": sorted(int(m) for m in serving["mnth"].unique()),
                "unaligned": {
                    "reference_rows": len(reference),
                    "feature_drift_share": round(unaligned.feature_drift_share, 4),
                    "target_drift": round(unaligned.target_drift.score, 4)
                    if unaligned.target_drift
                    else None,
                    "retrain": decide_retrain(unaligned).retrain,
                },
                "aligned": {
                    "reference_rows": len(comparable),
                    "feature_drift_share": round(aligned.feature_drift_share, 4),
                    "target_drift": round(aligned.target_drift.score, 4)
                    if aligned.target_drift
                    else None,
                    "method": aligned.target_drift.method if aligned.target_drift else None,
                    "retrain": decision.retrain,
                    "reason": decision.reason,
                },
            }
        )

    payload = {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": {
            "python": platform.python_version(),
            "river": __import__("river").__version__,
            "data_sha256": _digest_of_data(),
        },
        "setup": {
            "champion_trained_on_rows": cut,
            "control_rows": len(control),
            "control_mae": round(float(control.mean()), 2),
            "current_rows": len(treatment),
            "current_mae": round(float(treatment.mean()), 2),
            "note": (
                "Control is the held-out slice of 2011, the regime the champion was "
                "trained on. An alarm there is a false alarm."
            ),
        },
        "error_stream_at_defaults": at_defaults,
        "adwin_sweep": sweep,
        "adwin_calibrated": {
            "delta": chosen,
            "chosen_by": "most sensitive delta raising no alarm on the control",
            "current": calibrated.as_dict() if calibrated else None,
        },
        "window_monitor": window,
        "thresholds": {
            "target_threshold": CONFIG.drift.target_threshold,
            "feature_drift_share": CONFIG.drift.feature_drift_share,
            "min_window_rows": CONFIG.drift.min_window_rows,
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")

    print(
        f"\ncontrol MAE {payload['setup']['control_mae']}  "
        f"current MAE {payload['setup']['current_mae']}"
    )
    print("\nat default settings")
    for name, row in at_defaults.items():
        print(
            f"  {name:<12} control {row['control']['count']:>3} alarms (first "
            f"{row['control']['first']}) | 2012 {row['current']['count']:>3} alarms "
            f"(first {row['current']['first']})"
        )
    if chosen and calibrated:
        print(
            f"\nADWIN calibrated to delta={chosen}: 0 control alarms, "
            f"2012 first alarm at row {calibrated.first} (~{(calibrated.first or 0) / 24:.0f} days)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
