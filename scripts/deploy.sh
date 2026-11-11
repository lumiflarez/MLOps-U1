#!/usr/bin/env bash
set -euo pipefail

git pull --rebase
python scripts/check_model_gate.py
python scripts/register_model.py
kubectl apply -f k8s/api.yaml
kubectl rollout restart deployment/jailbreak-api -n mlops
kubectl rollout status deployment/jailbreak-api -n mlops