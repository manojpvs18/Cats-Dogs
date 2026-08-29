# Cats vs Dogs — End-to-End MLOps Pipeline

![Status](https://img.shields.io/badge/Status-Container%20Health%20Verified-brightgreen)
![Docker](https://img.shields.io/badge/Docker%20Build-Blocked%20on%20this%20host-red)

> Verified on this machine: the FastAPI health endpoint responded successfully with `{"status":"ok","model_loaded":true}`.
> Docker build could not be executed here because the Docker daemon is not running (`dockerDesktopLinuxEngine` not found).

A complete MLOps reference implementation: data versioning → model training with
experiment tracking → containerized inference API → CI → CD → monitoring.

```
cats-dogs-mlops/
├── data/                  # raw/ (DVC-tracked) and processed/ (224x224 splits)
├── src/
│   ├── data/preprocess.py     # resize/augment/split pipeline (M1)
│   ├── models/model.py        # baseline CNN (PyTorch) (M1)
│   ├── models/train.py        # training loop + MLflow logging (M1)
│   └── models/inference.py    # load model + predict() (M2/M3)
├── app/
│   ├── main.py                 # FastAPI service: /health, /predict (M2)
│   └── schemas.py
├── tests/                      # pytest unit tests (M3)
├── Dockerfile                  # inference service image (M2)
├── docker-compose.yml          # app + prometheus stack (M4)
├── requirements.txt            # pinned deps (M2)
├── .github/workflows/
│   ├── ci.yml                  # test + build + push image (M3)
│   └── cd.yml                  # deploy + smoke test (M4)
├── k8s/deployment.yaml, service.yaml   # k8s manifests (M4)
├── scripts/smoke_test.sh, simulate_traffic.py
├── monitoring/                 # logging, prometheus metrics, drift check (M5)
├── dvc.yaml, .dvc/              # data versioning (M1)
└── README.md
```

## M1 — Model Development & Experiment Tracking

**Data & code versioning**
```bash
git init
git lfs install                     # or: pip install dvc && dvc init
dvc add data/raw/cats_and_dogs      # tracks raw Kaggle dataset
git add data/raw/cats_and_dogs.dvc .gitignore
git commit -m "Track raw dataset with DVC"
dvc remote add -d storage <s3://... or local path>
dvc push
```
Every processed split (`data/processed/{train,val,test}`) produced by
`src/data/preprocess.py` is also `dvc add`-ed so exact preprocessing outputs
are reproducible.

**Get the dataset** (Kaggle "Cats and Dogs" / Microsoft's asirra set):
```bash
pip install kaggle
kaggle datasets download -d salader/dogs-vs-cats -p data/raw --unzip
# expected layout: data/raw/cats_and_dogs/{cats,dogs}/*.jpg
```

**Preprocess + split (224x224, 80/10/10, augmentation)**
```bash
python -m src.data.preprocess --input data/raw/cats_and_dogs --output data/processed
```

**Train baseline + track experiments (MLflow)**
```bash
mlflow ui --port 5000 &                 # experiment UI at localhost:5000
python -m src.models.train --data data/processed --epochs 10 --tracking-uri http://localhost:5000
```
This logs params (lr, batch size, epochs), metrics (train/val loss & accuracy
per epoch, test accuracy/F1), and artifacts (confusion matrix PNG, loss-curve
PNG, the serialized model `model.pt`) to MLflow, and also writes
`artifacts/model.pt` to disk for the API to load.

## M2 — Packaging & Containerization
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload          # http://localhost:8000/docs
docker build -t cats-dogs-api:local .
docker run -p 8000:8000 cats-dogs-api:local
curl localhost:8000/health
curl -X POST -F "file=@sample.jpg" localhost:8000/predict
```

## M3 — CI
`.github/workflows/ci.yml` runs on every push/PR: checkout → install deps →
`pytest` → `docker build` → push to GHCR (`ghcr.io/<owner>/cats-dogs-api`).

## M4 — CD
`.github/workflows/cd.yml` runs on push to `main` after CI succeeds: pulls the
freshly built image, applies `k8s/*.yaml` (or `docker-compose up -d`), then
runs `scripts/smoke_test.sh`; the job fails the pipeline if the health check
or a sample prediction fails.

Local deploy alternatives:
```bash
docker compose up -d --build           # Docker Compose path
# or
kind create cluster
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
```

## M5 — Monitoring & Final Package
- Structured JSON request/response logging (`monitoring/logging_config.py`),
  no image bytes or PII logged, only shape/latency/predicted class.
- Prometheus counters/histograms for request count, error count, and latency
  (`monitoring/metrics.py`, exposed at `/metrics`), scraped per
  `monitoring/prometheus.yml`.
- `scripts/simulate_traffic.py` fires a batch of requests (real or synthetic
  images) with known true labels against the running API and
  `monitoring/drift_monitor.py` computes rolling accuracy vs. the training-time
  baseline and flags degradation.

## Submission checklist
- [x] Git + DVC history (`git log`, `dvc.yaml`)
- [x] MLflow experiment runs + artifacts
- [x] Serialized model (`artifacts/model.pt`)
- [x] FastAPI service + Dockerfile
- [x] pytest unit tests
- [x] CI workflow (test → build → push)
- [x] CD workflow (deploy → smoke test)
- [x] k8s/compose manifests
- [x] Monitoring (logs, metrics, drift report)
