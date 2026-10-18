"""Send real rows from the dataset to a running service, to produce a prediction log.

`mlplatform monitor --source predictions` compares live serving traffic against
the training window, so it needs a prediction log to exist. Without one it exits
2 with "serve some traffic first", which is also what the `measure_drift` task in
the `demand_monitor` DAG hits. Nothing in the repository produced that log, so
this does.

The rows are real rows from `data/raw/hour.csv`, not invented ones: the point of
the drift check is that 2012 traffic differs from the 2011 the champion learned,
and made-up inputs would prove nothing about that. Defaults to the current
period (`yr == 1`, 2012).

    mlplatform serve &                      # or: docker compose up -d service
    python scripts/serve_traffic.py --rows 500

Prints what the service reported and where the log went. Needs a champion in the
registry; a service with no model answers 503 on /predict and this says so.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Exactly the fields service.Observation declares, in its order.
FIELDS = (
    "season",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
)
INTEGER_FIELDS = frozenset(FIELDS[:7])


def _post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000", help="service base URL")
    parser.add_argument("--rows", type=int, default=500, help="how many observations to send")
    parser.add_argument("--batch", type=int, default=100, help="observations per request")
    parser.add_argument(
        "--year",
        type=int,
        default=1,
        choices=(0, 1),
        help="dteday year flag: 0 is 2011 (the reference), 1 is 2012 (the current window)",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    import pandas as pd

    source = ROOT / "data" / "raw" / "hour.csv"
    if not source.exists():
        print(f"no dataset at {source}; run `mlplatform fetch` first", file=sys.stderr)
        return 2

    frame = pd.read_csv(source)
    frame = frame[frame["yr"] == args.year]
    if frame.empty:
        print(f"no rows with yr == {args.year}", file=sys.stderr)
        return 2

    rows = frame.head(args.rows)[list(FIELDS)]
    observations = []
    for record in rows.to_dict(orient="records"):
        observations.append(
            {k: (int(v) if k in INTEGER_FIELDS else float(v)) for k, v in record.items()}
        )

    sent = 0
    version = "unknown"
    for start in range(0, len(observations), args.batch):
        batch = observations[start : start + args.batch]
        try:
            body = _post(f"{args.url}/predict", {"observations": batch}, args.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            print(f"service returned {exc.code}: {detail}", file=sys.stderr)
            if exc.code == 503:
                print("no champion loaded; run `mlplatform train` first", file=sys.stderr)
            return 1
        except urllib.error.URLError as exc:
            print(f"could not reach {args.url}: {exc.reason}", file=sys.stderr)
            return 1

        got = len(body["predictions"])
        if got != len(batch):
            print(f"sent {len(batch)} observations, got {got} predictions", file=sys.stderr)
            return 1
        sent += got
        version = body.get("model_version", version)

    from mlplatform.config import CONFIG

    print(f"sent {sent} observations, model_version {version}")
    print(f"prediction log: {CONFIG.service.prediction_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
