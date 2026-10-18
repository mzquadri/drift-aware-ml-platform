# Airflow run logs

Raw stdout from the two `airflow dags test` runs described in
[`../AIRFLOW_RUN.md`](../AIRFLOW_RUN.md), kept as written rather than
summarised, so the claims in that document can be checked against the thing
they describe.

| File | Run |
|---|---|
| `demand_monitor.log` | `airflow dags test demand_monitor 2026-10-17` |
| `demand_train.log` | `airflow dags test demand_train 2026-10-17` |

Two lines carry most of the weight:

- `demand_monitor.log`: **`Branch into trigger_retrain`**. Before the artefact
  path was fixed, the branch read a file nothing wrote to and could only reach
  `no_action`.
- `demand_train.log`: **`challenger MAE 95.48 is not 2% better than champion
  95.48`**. The promotion gate refusing, with version 3 registered anyway and
  the alias left on version 2.

Every line appears twice. Airflow's task logger writes to both its own handler
and the console, and the duplication is in the captured output rather than
something added here.

An earlier `demand_monitor` attempt failed at `trigger_retrain` with
`DagNotFound: Dag id demand_train not found in DagModel`. That is an artefact of
`airflow dags test` running one DAG in isolation without registering its
siblings, not a defect in the DAG; `airflow dags reserialize` before the test is
the fix, and the log kept here is the run after it.
