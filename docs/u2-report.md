# U2 Report

## Architecture

```
Laptop:  dvc repro → metrics.json → git push
GitHub:  lint → test → quality-gate → build-push (GHCR) → bot updates k8s/api.yaml
Laptop:  deploy.sh → gate → @champion → kubectl apply + rollout restart
Cluster: Job (training) → MLflow (tracking + registry) → jailbreak-api (loads @champion)
```

| Part | Files |
|---|---|
| Training with tracking + HPO | `src/train.py`, `src/hpo.py` (Optuna, 20 nested runs) |
| Model registry | `scripts/register_model.py` (`jailbreak-detector`, alias `@champion`) |
| Training Job | `k8s/train-job.yaml`, `k8s/train-configmap.yaml`, Secret `mlflow-credentials` |
| Inference API | `src/serve.py`, `k8s/api.yaml` (Deployment ×2 + Service) |
| CI/CD | `.github/workflows/ci.yaml`, `scripts/check_model_gate.py`, `scripts/deploy.sh` |

## Measurements

| Metric | Value |
|---|---|
| p95 latency `/predict` | 12.03 ms |
| Throughput | 99.3 rps |
| Model size | 481 KB |
| Test accuracy (v1 `@champion`) | 0.973 |

Load test: 100 sequential requests via `kubectl port-forward`, 2 replicas (`docs/load-test.json`).
All NFR with measured values: `docs/requirements.md` v1.2.

## Quality gate

- **Green run** (CI 43022396546): gate passed (accuracy 0.973 ≥ 0.95) → image pushed, manifest updated.
- **Red run** (CI 43021556181): threshold 0.99 → `FAIL: accuracy=0.9730 < 0.99`, build-push skipped (`docs/red-run.log`).