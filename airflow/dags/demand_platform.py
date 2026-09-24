"""The two scheduled jobs, and the branch between them.

Both DAGs shell out to the same CLI a human uses. That is deliberate: if the
scheduled run and the manual run take different code paths, the one you debug is
never the one that broke.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

ARTIFACTS = Path("/opt/airflow/artifacts")

DEFAULT_ARGS = {
    "owner": "mzquadri",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,
}


with DAG(
    dag_id="demand_train",
    description="Fit a challenger and promote it only if it clears both gates",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule=None,  # triggered by the monitor, or by hand
    catchup=False,
    max_active_runs=1,
    tags=["ml", "training"],
) as train_dag:
    # --include-current is what makes a retrain worth doing: it folds the drifted
    # window into the training set instead of refitting on the same stale year.
    train = BashOperator(
        task_id="train_and_gate",
        bash_command="mlplatform train --include-current",
    )

    # The service caches the champion in memory, so a promotion is invisible
    # until it is told to look again.
    reload_service = BashOperator(
        task_id="reload_service",
        bash_command=(
            'curl -fsS -X POST "${SERVICE_URL:-http://service:8000}/reload" '
            "|| echo 'service unreachable, it will pick the model up on next start'"
        ),
    )

    train >> reload_service


def _branch_on_drift() -> str:
    """Read the monitor's own decision file rather than re-deriving it here.

    Two places deciding the same thing is how a pipeline ends up retraining when
    the dashboard says it should not have.
    """
    path = ARTIFACTS / "last_drift.json"
    if not path.exists():
        return "no_action"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return "trigger_retrain" if payload.get("retrain") else "no_action"


with DAG(
    dag_id="demand_monitor",
    description="Measure drift against the training window and retrain if it has moved",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule="0 3 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["ml", "monitoring"],
) as monitor_dag:
    measure = BashOperator(
        task_id="measure_drift",
        bash_command="mlplatform monitor --source predictions",
    )

    branch = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=_branch_on_drift,
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="demand_train",
        wait_for_completion=False,
    )

    no_action = EmptyOperator(task_id="no_action")

    measure >> branch >> [trigger_retrain, no_action]
