"""Export the MLflow tracking store to a tracked JSON record.

Why this exists: `mlflow.db`, `mlruns/` and `artifacts/` are all gitignored,
because they are reproducible outputs and a binary database is not reviewable in
a diff. But that left the repository claiming an MLflow integration with nothing
committed that shows it ever ran. This script closes that gap: it reads whatever
the tracking store actually holds and writes a readable record of it.

It reports what is there. It does not create runs, and it fails rather than
writing a file if the store is empty, so an empty export cannot be mistaken for
evidence of a run.

    MLFLOW_TRACKING_URI=sqlite:///mlflow.db python scripts/export_mlflow_runs.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "mlflow_training_runs.json"
DATA = ROOT / "data" / "raw" / "hour.csv"


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        print("MLFLOW_TRACKING_URI is not set", file=sys.stderr)
        return 2

    client = MlflowClient()
    from mlplatform.config import CONFIG

    experiment = client.get_experiment_by_name(CONFIG.mlflow.experiment)
    if experiment is None:
        print(f"no experiment named {CONFIG.mlflow.experiment!r} in {uri}", file=sys.stderr)
        return 1

    runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time ASC"])
    if not runs:
        print("the store holds no runs; refusing to write an empty record", file=sys.stderr)
        return 1

    versions = sorted(
        client.search_model_versions(f"name='{CONFIG.mlflow.registry_name}'"),
        key=lambda v: int(v.version),
    )

    try:
        champion = client.get_model_version_by_alias(
            CONFIG.mlflow.registry_name, CONFIG.service.model_alias
        )
        champion_version = champion.version
    except Exception:
        champion_version = None

    record = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": {
            "python": ".".join(str(n) for n in sys.version_info[:3]),
            "mlflow": mlflow.__version__,
            "tracking_store": uri.split("://", 1)[0],
            "data_sha256": _sha256(DATA),
        },
        "experiment": {"name": experiment.name, "id": experiment.experiment_id},
        "runs": [
            {
                "run_id": r.info.run_id,
                "status": r.info.status,
                "params": dict(sorted(r.data.params.items())),
                "metrics": {k: v for k, v in sorted(r.data.metrics.items())},
                "promoted": r.data.tags.get("promoted"),
                "promotion_reason": r.data.tags.get("promotion_reason"),
            }
            for r in runs
        ],
        "registry": {
            "name": CONFIG.mlflow.registry_name,
            "versions": [
                {"version": v.version, "run_id": v.run_id, "status": v.status} for v in versions
            ],
            "alias": CONFIG.service.model_alias,
            "alias_points_at": champion_version,
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(record['runs'])} run(s), "
          f"{len(versions)} registry version(s), alias -> {champion_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
