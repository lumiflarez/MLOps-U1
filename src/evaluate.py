import json
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

model = joblib.load("models/model.pkl")
test = pd.read_csv("data/processed/test.csv")
x_test, y_test = test["prompt"], test["label"]

preds = model.predict(x_test)

metrics = {
    "accuracy": accuracy_score(y_test, preds),
    "precision": precision_score(y_test, preds),
    "recall": recall_score(y_test, preds),
    "f1": f1_score(y_test, preds),
}

Path("metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2))

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
run_id = Path("models/run_id.txt").read_text().strip()
client = mlflow.MlflowClient()
for name, value in metrics.items():
    client.log_metric(run_id, f"test_{name}", value)

print(f"metrics attached to MLflow run {run_id}")