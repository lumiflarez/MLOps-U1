# MLOps U1 - Jailbreak Prompt Detection

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

## Ports and logins
| Service | Address | Login |
|---|---|---|
| MLflow UI | http://localhost:5000 (via port-forward) | none |
| PostgreSQL | inside the cluster only, `postgres:5432` | no manual login: MLflow connects automatically with the user/password from Secret `postgres-secret` |

## Where to see results
- MLflow UI, experiment `jailbreak-detection`: runs with parameters, metrics , tags, model
- `metrics.json`: test metrics of the current version
- `kubectl get all -n mlops`: MLflow and PostgreSQL in the cluster