# The Airflow run, and the three defects it found

`airflow/dags/demand_platform.py` had never been executed. The DAGs were
syntactically fine, the compose service looked complete, and nothing worked.
Running them found three defects, each of which only exists at run time and none
of which a reader would have spotted.

The most serious one meant **`demand_monitor` could never have triggered a
retrain** - the single behaviour the platform exists for.

## Defect 1: the branch read a file nothing wrote

`cli.py` wrote its decision to `ROOT/artifacts/last_drift.json`, where `ROOT` is
the installed package's parent. Inside the Airflow container that is
`/opt/airflow/project/artifacts/`. The DAG's branch read
`/opt/airflow/artifacts/last_drift.json`, a different directory on a different
mount.

The file was therefore never where the branch looked, and `_branch_on_drift`
takes its "no file" path when the decision is missing:

```python
if not path.exists():
    return "no_action"
```

So `demand_monitor` would have reached `no_action` on every run, for ever, with
every task green. A drifted model would never have been retrained and nothing
would have reported a problem.

This was also the real breach of what `config.py`'s own docstring promises -
"Nothing reads an environment variable anywhere else... which is the whole
reason training-serving skew is avoidable here" - because the DAG was spelling a
path itself instead of taking it from config.

**Fixed** by giving `Config` an `artifacts_dir`, read from `ARTIFACTS_DIR`, and
having both `cli.py` and the DAG use it. `tests/test_artifact_paths.py` holds
them to one directory, including a check that the DAG source contains no
hardcoded path, since a literal would pass every other assertion.

## Defect 2: Airflow had no database to start against

`docker-compose.yml` points Airflow at `postgres/airflow`:

```yaml
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://platform:platform@postgres/airflow
```

The `postgres` image creates only the database named in `POSTGRES_DB`, which is
`platform`. Nothing created `airflow`, so `airflow standalone` would have failed
at startup against a database that did not exist.

**Fixed** with `docker/postgres-init.sh`, mounted into
`/docker-entrypoint-initdb.d`. It runs on first boot only, so an existing
cluster needs `docker compose down -v` or a manual `CREATE DATABASE`.

## Defect 3: the artefact volume was not writable by Airflow

The `artifacts` volume is one volume mounted in two places: `/app/artifacts` in
the service, `/opt/airflow/artifacts` in Airflow. Docker initialises a new named
volume from whatever the image has at the mount point, ownership included - and
neither image had that directory, so the volume was created `root:root 0755`
while the Airflow container runs as uid 50000:

```
drwxr-xr-x 2 0 0 4096 /opt/airflow/artifacts
touch: cannot touch '/opt/airflow/artifacts/.probe': Permission denied
```

`mlplatform monitor` could not write `last_drift.json`, so `measure_drift` would
have failed on a permission error that says nothing about drift.

**Fixed** by creating the directory `0777` in both `docker/Dockerfile.airflow`
and `docker/Dockerfile`, so the result does not depend on which container starts
first. World-writable rather than owned by one user because two images with
different uids share the volume; that is the simplest correct answer for a
single-machine compose stack and not a pattern for a real deployment.

## Two further things the compose file had wrong

Both inherited from the shared environment block, which spells the service's own
`/app` paths. The Airflow container has no `/app`.

| Variable | Was | Now |
|---|---|---|
| `PREDICTION_LOG` | `/app/artifacts/predictions.parquet` | `/opt/airflow/artifacts/predictions.parquet` |
| `ARTIFACTS_DIR` | not present | `/opt/airflow/artifacts` |

Left as inherited, `mlplatform monitor --source predictions` read a prediction
log under a directory that does not exist in that container.

## The image is now built, not installed on every boot

The service used to run:

```yaml
command: bash -c "pip install --user -e /opt/airflow/project && airflow standalone"
```

That reinstalls the project every time the container starts, needs the network
every time, and leaves the installed versions depending on when the container
last booted rather than on anything recorded. It is now
`docker/Dockerfile.airflow`, so the install happens once and is layer-cached.

**One dependency is resolved against Airflow's own constraints, deliberately.**
Installing the project under the 2.10.2 constraints file is impossible: it pins
`pandas==2.1.4` and this project needs `pandas>=2.2`. Checking what Airflow
actually requires:

```
$ pip show apache-airflow -> 2.10.2
pandas requirement lines: ["pandas>=1.2.5,<2.2; extra == 'pandas'",
                           "pandas>=1.2.5,<2.2; extra == 'devel-ci'"]
```

Core Airflow does not depend on pandas at all - that bound belongs to two
optional extras, neither of which is installed - and the DAGs use only Bash,
Branch, Empty and TriggerDagRun operators, none of which touch pandas. So the
project's requirement is allowed to win and pandas is upgraded. If a provider
that does use pandas is ever added, the fix is to move Airflow forward rather
than pandas back, because the training code is what needs 2.2.

## What was actually run

Airflow is pinned to `apache-airflow==2.10.2` in the `airflow` extra of
`pyproject.toml`, matching the compose image.

```sh
docker compose build airflow
docker compose down -v                 # so the postgres init script runs
docker compose up -d postgres
docker compose run --rm airflow airflow db migrate
docker compose run --rm airflow airflow dags reserialize
docker compose run --rm \
  -e MLFLOW_TRACKING_URI="sqlite:////opt/airflow/project/mlflow.db" \
  airflow airflow dags test demand_monitor 2026-10-17
```

`airflow dags test` executes the DAG's tasks in process. It is used instead of
`airflow standalone` because this machine has 7.7 GB of RAM and could not hold a
webserver, scheduler and triggerer alongside PostgreSQL; several containers were
OOM-killed (`Exited (137)`) while trying. The tasks themselves are real: the
BashOperators shell out to the same `mlplatform` CLI a human runs, which is the
DAG file's own stated design.

`airflow dags reserialize` is needed before the test because `dags test` runs one
DAG in isolation and does not register its siblings. Without it,
`TriggerDagRunOperator` fails with `DagNotFound: Dag id demand_train not found in
DagModel`, which is an artefact of the test harness rather than a defect in the
DAG.

### Both DAGs parse in the container

```
dag_id         | fileloc                              | owners   | is_paused
===============+======================================+==========+==========
demand_monitor | /opt/airflow/dags/demand_platform.py | mzquadri | None
demand_train   | /opt/airflow/dags/demand_platform.py | mzquadri | None

$ airflow dags list-import-errors
No data found
```

### demand_monitor took the branch that used to be unreachable

The prediction log is 4,500 real observations, drawn from `data/raw/hour.csv` and
sent through the running service by `scripts/serve_traffic.py`, then copied into
the shared volume.

```
{"retrain": true,
 "reason": "target 'cnt' drifted 0.406 by Wasserstein distance (normed),
            over the 0.2 threshold, while only 45% of features moved;
            the relationship the model learned no longer holds"}

Marking task as SUCCESS. task_id=measure_drift
Done. Returned value was: trigger_retrain
Branch into trigger_retrain
Following branch {'trigger_retrain'}
Skipping tasks [('no_action', -1)]
Marking task as SUCCESS. task_id=branch_on_drift
```

`trigger_retrain`, not `no_action`. That line is the whole point of defect 1:
before the fix the branch read a path nothing wrote to, so this run would have
skipped `trigger_retrain` and reported success while doing nothing.

The decision is label-shift rather than feature drift - the target moved 0.406
while only 45% of features did - which is the case `decide_retrain` is written to
catch and the reason it compares the target separately.

`trigger_retrain` then succeeded, and the whole run finished green.

### demand_train trained, and the gate refused

```sh
docker compose run --rm \
  -e MLFLOW_TRACKING_URI="sqlite:////opt/airflow/project/mlflow.db" \
  airflow airflow dags test demand_train 2026-10-17
```

```
promotion: challenger MAE 95.48 is not 2% better than champion 95.48
           (needed <= 93.57)
Registered model 'bike-demand' already exists. Creating a new version...
Created version '3' of model 'bike-demand'.
summary written to /opt/airflow/artifacts/last_training.json
          (mlflow run 258a3f9fb10a47fe...)
Marking task as SUCCESS. task_id=train_and_gate

Running command: curl -fsS -X POST ".../reload" || echo 'service unreachable...'
service unreachable, it will pick the model up on next start
Marking task as SUCCESS. task_id=reload_service

DagRun Finished: dag_id=demand_train
```

Three things are worth reading off that.

The **gate refused**, because the DAG trained on the same window the champion
already held and matched its 95.48 MAE exactly. Version 3 was registered anyway
and the `champion` alias stayed on version 2, which is what `log_run` promises.
`results/mlflow_training_runs.json` now holds all three runs, and between them
they cover every branch of the promotion decision: no champion, a challenger
that clears the margin, and one that does not.

The summary went to **`/opt/airflow/artifacts/`**, the shared volume, rather than
under the project root. That is defect 1 and defect 3 fixed, visible in one line.

The MLflow run landed in the **same tracking store as the two runs made by hand
on the host**, through the bind-mounted project directory, so the DAG's run and
the manual runs are in one experiment rather than two.

## Honest limits of this run

- **`airflow standalone` was not run.** The scheduler and webserver together do
  not fit on this machine. What is shown is the DAGs parsing and their tasks
  executing, not a scheduled run picked up by a live scheduler.
- **`reload_service` runs against no service.** Its command ends in
  `|| echo 'service unreachable...'`, so it passes either way. That tolerance is
  the DAG's own design, not something added here.
- **MLflow tracking points at SQLite**, in the bind-mounted project directory,
  for the same memory reason described in `MLFLOW_RUN.md`. The compose server
  is defined and its version mismatch fixed, but it was not up during this run.
- **The 0.406 drift figure depends on the traffic that was sent.** 4,500 rows of
  2012 across six months; a different slice gives a different number, and with
  only one month the monitor declines to decide at all because fewer than 1,000
  reference rows cover the seasons being served.
