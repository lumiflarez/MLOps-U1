# U1 Report - Jailbreak Prompt Detection

## What was done
| Requirement | Result |
|---|---|
| R1 Repo + requirements | `src/`, `docker/`, `k8s/`, `docs/`; README ; `docs/requirements.md` |
| R2 Pipeline | `dvc.yaml` prepare -> train -> evaluate, `params.yaml`, `dvc.lock` committed, local DVC remote; 3 MLflow runs with different hyperparameters; `reproduce.py` restored a run in 8.6 s with metric delta 0.0000 |
| R3 Docker | `docker/Dockerfile.train`, published as `ghcr.io/lumiflarez/mlops-u1-train:sha-dc34993`, 950 MB, Trivy: 0 CRITICAL, 48 HIGH, `docs/docker.md` |
| R4 Cluster + MLflow | kind cluster, `mlops` namespace, PostgreSQL database that stores run information (parameters, metrics, tags), MLflow server that stores model files on its own disk (PVC), access via port-forward, all runs logged from the laptop to the cluster MLflow |

Experiments:

| Run | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (C=1.0, max_features=5000) | 0.961 | 0.992 | 0.931 | 0.961 |
| max_features=50 | 0.903 | 0.942 | 0.863 | 0.900 |
| C=0.01 | 0.946 | 0.953 | 0.939 | 0.946 |

## What did not work (and fixes)
- **kind cluster creation failed:** Docker on WSL 2 used cgroup v1, which recent Kubernetes does not support.
  Fix: `kernelCommandLine = cgroup_no_v1=all` in `.wslconfig` and `wsl --shutdown`.
- **MLflow could not connect to PostgreSQL:** the official MLflow image has no PostgreSQL driver.
  Fix: image with `psycopg2-binary`, loaded into the node with `kind load`.
- **MLflow pod OOMKilled** at a 1 GiB memory limit. Fix: `--workers=1` and a 2 GiB limit.
- **NFR-01 not met:** recall 0.931 < 0.95.

## Time spent
| Part | Hours |
|---|---|
| Tools, repo, data, DVC | 2 |
| Kubernetes cluster, PostgreSQL, MLflow | 5 |
| Pipeline, experiments, reproduce | 4 |
| Docker image, GHCR, Trivy | 2 |
| Documentation | 2 |
| **Total** | **15** |