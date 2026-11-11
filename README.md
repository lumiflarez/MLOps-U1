# MLOps U1+U2 - Jailbreak Prompt Detection

## Run from scratch

### Prerequisites
Docker, kind v0.33.0, kubectl, Python 3.10, Git.

### 1. Python environment
    git clone https://github.com/lumiflarez/MLOps-U1.git && cd MLOps-U1
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

### 2. Data
The DVC remote in `.dvc/config` is a local folder on the author's machine, so download the raw
data from Hugging Face:

    mkdir -p data/raw && python src/download_data.py

### 3. Cluster with MLflow
    kind create cluster --config k8s/kind-config.yaml
    kubectl apply -f k8s/namespace.yaml

    kubectl create secret generic postgres-secret -n mlops \
      --from-literal=POSTGRES_USER=mlflow \
      --from-literal=POSTGRES_PASSWORD=<choose-a-password> \
      --from-literal=POSTGRES_DB=mlflow
    kubectl apply -f k8s/postgres.yaml

    docker build -f docker/Dockerfile.mlflow -t mlflow-server:3.15.2 docker/
    kind load docker-image mlflow-server:3.15.2 --name mlops-u1
    kubectl apply -f k8s/mlflow.yaml

    kubectl get pods -n mlops -w                         # wait until both pods are 1/1 Running
    kubectl port-forward -n mlops svc/mlflow 5000:5000   # in a second terminal and keep this terminal open

### 4. Pipeline
    dvc repro                      # prepare -> train -> evaluate, run logged to cluster MLflow
    python reproduce.py <run_id>   # reproduce a run; take the run ID from your MLflow UI

### 5. Hyperparameter search (Optuna) and model registry
    python src/hpo.py                      # 20 Optuna trials as nested MLflow runs under "optuna-hpo"
    # copy best params from the "optuna-hpo" run into params.yaml, then:
    dvc repro                              # final model with the best params
    python scripts/register_model.py       # registers the model as "jailbreak-detector" and sets @champion

### 6. Training Job in the cluster
    GIT_SHA=$(git rev-parse --short=8 HEAD)
    docker build -f docker/Dockerfile.train --build-arg GIT_COMMIT=$GIT_SHA -t jailbreak-train:sha-$GIT_SHA .
    kind load docker-image jailbreak-train:sha-$GIT_SHA --name mlops-u1
    # set the same tag in k8s/train-job.yaml (image: jailbreak-train:sha-...)

    kubectl create secret generic mlflow-credentials -n mlops \
      --from-literal=MLFLOW_TRACKING_USERNAME=trainer \
      --from-literal=MLFLOW_TRACKING_PASSWORD=<choose-a-password>

    kubectl apply -f k8s/train-configmap.yaml
    kubectl delete job jailbreak-train -n mlops --ignore-not-found #delete old job is there is one
    kubectl apply -f k8s/train-job.yaml
    kubectl logs -f job/jailbreak-train -n mlops
    kubectl get job jailbreak-train -n mlops      # STATUS Complete, COMPLETIONS 1/1

    # To train with different hyperparameters, edit `params.yaml`, regenerate the ConfigMap and rerun the Job
    kubectl create configmap train-config -n mlops --from-file=params.yaml \
      --from-literal=MLFLOW_TRACKING_URI=http://mlflow.mlops.svc:5000 \
      --dry-run=client -o yaml > k8s/train-configmap.yaml
    kubectl apply -f k8s/train-configmap.yaml
    kubectl delete job jailbreak-train -n mlops --ignore-not-found
    kubectl apply -f k8s/train-job.yaml

### 7. Inference API in the cluster
The API loads the model from the MLflow registry by alias `@champion`.

    kubectl apply -f k8s/api.yaml
    kubectl rollout status deployment/jailbreak-api -n mlops
    kubectl port-forward -n mlops svc/jailbreak-api 8080:80     # keep this terminal open

    curl -s localhost:8080/health
    curl -s -X POST localhost:8080/predict -H "Content-Type: application/json" \
      -d '{"prompt": "Ignore all previous instructions"}'
    python scripts/load_test.py                                  # p95 and throughput 

### 8. CI/CD
On every push to `main`, GitHub Actions (`.github/workflows/ci.yaml`) runs:
lint → tests → quality gate → build and push the API image to GHCR (tagged `sha-<commit>`)
→ update the image in `k8s/api.yaml`.
The quality gate checks that the accuracy in `metrics.json` is at least `gate.threshold` from `params.yaml`.

After a green run, deploy to the local cluster (needs the MLflow port-forward and `MLFLOW_TRACKING_URI`):

    git pull --rebase
    ./scripts/deploy.sh          # gate -> register model as @champion -> kubectl apply -> rollout restart

Rollback to a previous model version:

    python scripts/register_model.py --run-id <run id of the previous version>
    kubectl rollout restart deployment/jailbreak-api -n mlops

## Ports and logins
| Service | Address | Login |
|---|---|---|
| MLflow UI | http://localhost:5000 (via port-forward) | none |
| PostgreSQL | inside the cluster only, `postgres:5432` | no manual login: MLflow connects automatically with the user/password from Secret `postgres-secret` |
| Inference API | http://localhost:8080 (via port-forward), interactive docs at /docs | none |

## Where to see results
- MLflow UI, experiment `jailbreak-detection`: runs with parameters, metrics, tags, model
- `metrics.json`: test metrics of the current version
- `kubectl get all -n mlops`: MLflow and PostgreSQL in the cluster
- MLflow UI, run `optuna-hpo`: expand it to see the nested trial runs (HPO tree)
- MLflow UI → Model registry → `jailbreak-detector`: versions and the `@champion` alias
- `kubectl get job,pods -n mlops -l app=jailbreak-train`: training Job status
- `docs/train-job.log`: saved log of a successful training Job
- `curl localhost:8080/health`: model version currently served
- GitHub → Actions: CI/CD runs 
- GitHub → Packages: `jailbreak-api` images tagged `sha-<commit>`
- `docs/load-test.json`: p95 latency and throughput
- `docs/requirements.md` (v1.2, measured NFR), `docs/u2-report.md`