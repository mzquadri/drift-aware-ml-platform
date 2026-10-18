# The MLflow run this repository is evidenced by

Until this run, the MLflow integration in `src/mlplatform/registry.py` was code
that had never been executed. The README described a promotion sequence, but
nothing committed showed a run, a registered version or an alias move. This file
records the run that fixed that, and the two defects the run exposed.

Committed evidence: [`results/mlflow_training_runs.json`](../results/mlflow_training_runs.json),
written by [`scripts/export_mlflow_runs.py`](../scripts/export_mlflow_runs.py)
straight from the tracking store. `mlflow.db`, `mlruns/` and `artifacts/` stay
gitignored, so that export is the reviewable record.

## Two defects the run exposed

Both were invisible while nothing ran against the server.

**1. The client and the server could not have worked together.** `pyproject.toml`
asked for `mlflow>=3.0`, while `docker-compose.yml` ran the server at
`v2.16.2`. `registry.py` calls `mlflow.sklearn.log_model(name=...)` and passes
`skops_trusted_types`, both MLflow 3 forms that a 2.x server has no route for.
The client is now pinned to `mlflow==3.16.1` and the server image to
`v3.16.1`, so the two move together.

**2. The server could not open its own backend store.** The compose service
points `--backend-store-uri` at PostgreSQL, but the official
`ghcr.io/mlflow/mlflow` image does not ship `psycopg2`. The command now
installs the driver before starting the server.

## What was run

```sh
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"

.venv/Scripts/python -m mlplatform.cli fetch          # UCI Bike Sharing, SHA-256 checked

export MLFLOW_TRACKING_URI="sqlite:///$(pwd)/mlflow.db"
.venv/Scripts/python -m mlplatform.cli train                    # first model
.venv/Scripts/python -m mlplatform.cli train --include-current  # challenger
.venv/Scripts/python -m mlplatform.cli monitor                  # drift decision
.venv/Scripts/python scripts/export_mlflow_runs.py
```

`pytest` passes 48 tests against this tree.

The `monitor` step returned `retrain: true`, with the target `cnt` drifted
**0.679** by normed Wasserstein distance while only 9% of features moved. That
0.679 is the same figure the README's sensitivity table already carried for the
full 8,734-row window, so the committed table and a fresh run agree.

Data is the UCI Bike Sharing Dataset, fetched by `mlplatform fetch` from
`archive.ics.uci.edu`, SHA-256
`e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f` - the same
hash `results/drift_latency.json` recorded, so both results describe one dataset.

### The tracking store is SQLite, which is this repository's documented local route

The run used a local SQLite tracking store. That is not a workaround invented
for it: it is the line the README's own "Run it" section gives,
`export MLFLOW_TRACKING_URI=sqlite:///mlflow.db   # registry without a server`,
and the commands above are that quickstart in order. A database-backed store is
all the model registry and aliases in `registry.py` require; a server is not.

The compose server was still brought up first, because the two fixes above
needed checking against the thing they fix. It got as far as
`Creating initial MLflow database tables...` against PostgreSQL, which is what
shows the pinning and the `psycopg2` install are both right. It then could not
hold a worker process on this machine - 7.7 GB of RAM, and the server's children
were killed repeatedly (`Child process [N] died`) under memory pressure - so the
run was completed on the documented SQLite route instead.

What this run therefore does and does not show:

| Shown | Not shown |
|---|---|
| Runs logged with parameters, metrics and tags | The PostgreSQL backend store serving a full run |
| A model registered, versioned twice | Artifact serving through `--serve-artifacts` |
| The `champion` alias created and then moved | Multiple clients against one shared server |
| The promotion gate refusing and accepting on real metrics | |

## What the three runs recorded

All three are in the export with their full parameter sets. The third was run by
Airflow rather than by hand; see [`AIRFLOW_RUN.md`](AIRFLOW_RUN.md).

| | Run 1 | Run 2 (`--include-current`) | Run 3 (via Airflow) |
|---|---|---|---|
| Run id | `b894916c220e460f` | `f5b6a0999a874f3d` | `258a3f9fb10a47fe` |
| Validation rows | 1,729 | 3,476 | 3,476 |
| Validation MAE | 51.49 | 95.48 | 95.48 |
| Validation RMSE | 73.50 | 135.49 | 135.49 |
| Validation R-squared | 0.6211 | 0.6224 | 0.6224 |
| Seasonal-naive baseline MAE | 98.94 | 174.98 | 174.98 |
| Registry version | 1 | 2 | 3 |
| Promoted | yes | yes | **no** |

Run 3 is the one that shows the gate refusing. It trained on the same window as
run 2 and scored the same 95.48 MAE, so the margin test failed:

```
promotion: challenger MAE 95.48 is not 2% better than champion 95.48
           (needed <= 93.57)
```

**Version 3 was still registered, and the alias stayed on version 2.** That is
the behaviour `log_run` documents - "every run is registered, so a refused
challenger stays inspectable. Only a promoted one gets the alias moved onto it" -
and the export shows it directly: three versions, `champion` pointing at 2.

Between them the three runs exercise all three branches of the promotion
decision: no champion yet, a challenger that clears the margin, and a challenger
that does not.

Run 1's MAE of 51.49 is the same figure `results/drift_latency.json` carries as
`control_mae`, which is a useful cross-check: that file was produced by a
separate experiment script, and the training pipeline reproduced its number.

Run 2 is the interesting one, because it is the gate doing actual work rather
than waving through a first model. Training on the reference window plus the
drifted current window, the existing champion was re-scored on the new
validation window and came back at **116.57 MAE**. The challenger scored
**95.48**, which clears the 2% promotion margin, so the alias moved to version 2.

The MAE rising from 51.49 to 95.48 between the runs is not a regression. The two
numbers are measured on different validation windows: run 2's window includes
2012, which has both higher ridership and the drift the platform exists to
detect. The comparison that decides promotion is challenger against champion **on
the same window** - 95.48 against 116.57 - and that is the one the gate used.

## Reproducing it

`scripts/export_mlflow_runs.py` refuses to write a file when the store holds no
runs, so an empty export cannot be mistaken for evidence. Re-running `train`
appends versions rather than replacing them, so a fresh store is the way to
reproduce the table above exactly:

```sh
rm -f mlflow.db && rm -rf mlartifacts
```
