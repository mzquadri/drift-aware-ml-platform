# drift-aware-ml-platform

An hourly demand-forecasting service that notices when its own model has gone
stale, and retrains itself when it has.

Most portfolio MLOps projects train a model, wrap it in FastAPI and stop. The
interesting problems start after that: how do you know the model is still any
good, what stops a worse model from shipping, and what happens when the world
moves. This project is built around those three questions.

## What it actually does

```
                       ┌──────────────┐
  UCI hourly data ───► │ train (gate) │ ──► MLflow registry ──► champion
                       └──────────────┘            ▲
                              ▲                    │ promote only if better
                              │ retrain            │
                       ┌──────────────┐            │
                       │ drift check  │ ◄── serving traffic log
                       └──────────────┘            ▲
                                                   │
  request ──► FastAPI service ──► prediction ──────┘
                    │
                    └──► Prometheus ──► Grafana
```

Airflow runs the two jobs on a schedule. The monitor branches: if the serving
window has not drifted it does nothing, and if it has it triggers the training
DAG, which is the only thing allowed to promote a model.

## The three decisions that make it a platform

**A model only ships if it is measurably better.** Two gates, both in
[`train.py`](src/mlplatform/train.py). A challenger must beat "predict the
training mean" by 35%, which catches a broken dataset or a model that failed to
learn. Then it must beat the current champion by 2%, which stops a
noise-better model from churning production every night. Both thresholds are in
config, and the reason a model was refused is written to the run summary, so a
failed promotion is auditable rather than mysterious.

**Drift is measured and acted on separately.** Evidently produces the
statistics; [`decide_retrain`](src/mlplatform/drift.py) applies the policy. That
split means the policy is unit-testable without running a report, and the
awkward cases live somewhere you can see them. The one that bites people: a
small serving window fails a distribution test for reasons that have nothing to
do with the world changing, so the size guard runs *before* the drift check and
wins regardless of how dramatic the measured shift is.

**Liveness and readiness are different questions.** `/health` is true whenever
the process can answer, and deliberately does not depend on the model. If it
did, a registry outage would turn into a restart loop that cannot possibly fix
itself. `/ready` is true only once a champion is loaded, so traffic is not
routed to an instance that would answer with nothing.

## Why this dataset

[UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset),
hourly counts for 2011 and 2012. Training on 2011 and serving 2012 is not an
artificial split: ridership grew substantially between the two years, so the
serving window really is drifted relative to the training window. The drift
monitor has something true to find rather than noise dressed up as a demo.

`casual` and `registered` sum to `cnt` exactly, so both are dropped in
[`data.py`](src/mlplatform/data.py) before anything else sees the frame. A model
handed either one scores almost perfectly in validation and is worthless in
production. Dropping them once, at the edge, is why no later stage has to
remember to.

The validation split is the chronological tail, not a random sample. Demand is a
time series; a random split lets the model validate against hours whose
neighbours it trained on.

## Measured results

Reference period (2011), held-out tail of 1,729 hours:

| Metric | Model | Mean baseline |
|---|---|---|
| MAE | **51.49** | 98.94 |
| RMSE | **73.50** | — |
| R² | **0.621** | 0.0 |

The model beats the mean baseline by 48%, which clears the 35% gate, so the
first run promotes. Reproduce with `mlplatform train`; the run summary lands in
`artifacts/last_training.json`.

## Running it

Everything local, no cloud account needed:

```bash
pip install -e ".[dev]"
mlplatform fetch          # download and pin the dataset by hash
mlplatform train          # fit a challenger, apply both gates
mlplatform monitor        # compare 2012 against 2011, decide on retraining
mlplatform serve          # http://localhost:8000/docs
```

The full stack, with MLflow, Airflow, Prometheus and Grafana:

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Prediction API | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8080 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

## Deploying it

[`infra/terraform`](infra/terraform) provisions the service on Google Cloud Run
with a GCS bucket for artifacts and Artifact Registry for images. It is a small
module on purpose; the point is that the deployment is described in code and
reproducible, not that it is a large one.

```bash
cd infra/terraform
terraform init
terraform apply -var project_id=YOUR_PROJECT
```

## Tests

21 tests, none of which need a network, a registry or a trained model. They
cover the parts where mistakes are expensive: the promotion gate, the drift
policy, and what the service does before a model exists.

```bash
pytest
```

## What this is not

A portfolio project, run against a public dataset on one machine. It is not
production experience and is not presented as any. What it is meant to show is
the reasoning: which failure modes are worth engineering against, and what the
trade-off was each time.

Known gaps, stated rather than hidden: the Terraform module has been applied
against a single project and is not hardened; there is no feature store; the
Grafana dashboard covers service and prediction metrics but not per-feature
distributions; and retraining uses a full refit rather than anything incremental.

## Licence

MIT. The dataset is redistributed by UCI under its own terms and is downloaded
at runtime, not vendored.
