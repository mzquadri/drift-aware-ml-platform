"""The CLI and the Airflow DAG must agree on where artefacts live.

These exist because they did not agree, and nothing caught it. `cli.py` wrote
`last_drift.json` under `ROOT/artifacts`, which inside the Airflow container is
`/opt/airflow/project/artifacts`. The DAG's branch read `/opt/airflow/artifacts`,
a different directory on a different mount. The file was therefore never where
the branch looked, `_branch_on_drift` fell through to its "no file" case on every
run, and `demand_monitor` would have taken the `no_action` edge forever - a
drifted model would never have been retrained, which is the one thing the
platform is for.

A hardcoded path in the DAG was also the actual breach of what `config.py`'s
docstring promises: that nothing outside it reads the environment, so the
training run, the service and the Airflow tasks all see one set of values.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path


def _reloaded_config(monkeypatch, value: str | None):
    """Re-import config with ARTIFACTS_DIR set, since the dataclass reads it at import."""
    if value is None:
        monkeypatch.delenv("ARTIFACTS_DIR", raising=False)
    else:
        monkeypatch.setenv("ARTIFACTS_DIR", value)
    import mlplatform.config as config

    return importlib.reload(config)


def test_artifacts_dir_defaults_under_the_project_root(monkeypatch):
    config = _reloaded_config(monkeypatch, None)
    assert config.CONFIG.artifacts_dir == config.ROOT / "artifacts"


def test_artifacts_dir_is_settable_from_the_environment(monkeypatch, tmp_path):
    config = _reloaded_config(monkeypatch, str(tmp_path / "elsewhere"))
    assert config.CONFIG.artifacts_dir == tmp_path / "elsewhere"


def test_the_cli_writes_the_drift_decision_where_config_says(monkeypatch, tmp_path):
    """The regression guard: the CLI's write location is the configured directory.

    Asserted against the configured value rather than a literal, so pointing
    ARTIFACTS_DIR at the Airflow mount makes the CLI write there too.
    """
    target = tmp_path / "airflow-artifacts"
    config = _reloaded_config(monkeypatch, str(target))

    import mlplatform.cli as cli

    importlib.reload(cli)

    assert cli.ARTIFACTS == config.CONFIG.artifacts_dir == target


def test_the_dag_resolves_the_same_decision_file_as_the_cli(monkeypatch, tmp_path):
    """What the DAG's branch reads must be what the CLI wrote.

    The DAG module itself imports Airflow, which is not a test dependency, so
    this checks the contract both sides rely on: one configured directory, and
    `last_drift.json` inside it.
    """
    target = tmp_path / "shared-artifacts"
    config = _reloaded_config(monkeypatch, str(target))

    import mlplatform.cli as cli

    importlib.reload(cli)

    from mlplatform.drift import DriftResult, RetrainDecision, write_decision

    written = write_decision(
        RetrainDecision(retrain=True, reason="target moved", drift=DriftResult()),
        cli.ARTIFACTS / "last_drift.json",
    )

    dag_side = config.CONFIG.artifacts_dir / "last_drift.json"
    assert dag_side == written
    assert json.loads(dag_side.read_text(encoding="utf-8"))["retrain"] is True


def test_the_dag_does_not_hardcode_an_artifacts_path():
    """The DAG must take the directory from config, not spell a path itself.

    A literal here is what the bug was, and a literal would pass every test
    above while still disagreeing with the CLI in the container.
    """
    dag = Path(__file__).resolve().parents[1] / "airflow" / "dags" / "demand_platform.py"
    source = dag.read_text(encoding="utf-8")

    # Comments are stripped first. The path is named in a comment there on
    # purpose, to record what the bug was, and prose explaining a fix must not
    # read as the fix being absent.
    code = "\n".join(
        line.split("#", 1)[0] for line in source.splitlines() if not line.lstrip().startswith("#")
    )

    assert "/opt/airflow/artifacts" not in code
    assert "CONFIG.artifacts_dir" in code
