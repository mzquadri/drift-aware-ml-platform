"""Command line entry points, so Airflow and a human run exactly the same code."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import CONFIG, ROOT

log = logging.getLogger("mlplatform")

ARTIFACTS = ROOT / "artifacts"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    )


def cmd_train(args: argparse.Namespace) -> int:
    """Fit a challenger and register it only if it clears both gates."""
    from .data import period_frames
    from .registry import log_run
    from .train import run_training, write_run_summary

    periods = period_frames(CONFIG.data)
    frame = periods.reference
    if args.include_current:
        import pandas as pd

        frame = pd.concat([periods.reference, periods.current], ignore_index=True)
        log.info("training on reference + current (%d rows)", len(frame))

    run = run_training(frame, CONFIG)
    summary = write_run_summary(run, ARTIFACTS / "last_training.json")

    run_id = log_run(run, CONFIG, extra_params={"include_current": args.include_current})
    log.info("summary written to %s (mlflow run %s)", summary, run_id or "not tracked")

    print(json.dumps(run.decision.as_dict(), indent=2))
    # A run that trains a model but refuses to ship it is a success, not a failure.
    # Exit non-zero only when the model could not even beat the mean.
    return 0 if run.decision.promote or run.decision.champion_mae is not None else 1


def cmd_monitor(args: argparse.Namespace) -> int:
    """Compare the serving window against the training window and decide."""
    from .data import period_frames
    from .drift import decide_retrain, measure, write_decision, write_report
    from .features import feature_frame

    periods = period_frames(CONFIG.data)
    reference = feature_frame(periods.reference, CONFIG)

    if args.source == "periods":
        current = feature_frame(periods.current, CONFIG)
    else:
        import pandas as pd

        path = CONFIG.service.prediction_log
        if not path.exists():
            log.error("no prediction log at %s; serve some traffic first", path)
            return 2
        current = feature_frame(pd.read_parquet(path), CONFIG)

    result = measure(reference, current, CONFIG)
    decision = decide_retrain(result, CONFIG)

    write_decision(decision, ARTIFACTS / "last_drift.json")
    if not args.no_html:
        report = write_report(reference, current, ARTIFACTS / "drift_report.html", CONFIG)
        log.info("report written to %s", report)

    print(json.dumps({"retrain": decision.retrain, "reason": decision.reason}, indent=2))
    # Exit 0 either way: "no drift" is a normal outcome, not an error. The
    # Airflow branch reads the JSON rather than the exit code.
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "mlplatform.service:app",
        host=CONFIG.service.host,
        port=args.port or CONFIG.service.port,
        reload=args.reload,
    )
    return 0


def cmd_fetch(_: argparse.Namespace) -> int:
    from .data import ensure_raw

    path: Path = ensure_raw(CONFIG.data)
    print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlplatform", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download and pin the dataset")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("train", help="train a challenger and gate its promotion")
    p.add_argument(
        "--include-current",
        action="store_true",
        help="train on the reference period plus the drifted current period",
    )
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("monitor", help="measure drift and decide whether to retrain")
    p.add_argument(
        "--source",
        choices=["periods", "predictions"],
        default="periods",
        help="compare against the second year, or against real logged traffic",
    )
    p.add_argument("--no-html", action="store_true", help="skip the Evidently HTML report")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("serve", help="run the prediction service")
    p.add_argument("--port", type=int)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
