import argparse
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient

MODEL_NAME = "jailbreak-detector"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    client = MlflowClient()

    run_id = args.run_id or Path("models/run_id.txt").read_text().strip()

    existing = client.search_model_versions(f"name='{MODEL_NAME}' and run_id='{run_id}'")
    if existing:
        version = existing[0].version
        print(f"jau uzregistruota: {MODEL_NAME} v{version} (run {run_id})")
    else:
        mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        version = mv.version
        print(f"uzregistruota: {MODEL_NAME} v{version} (run {run_id})")

    run_metrics = client.get_run(run_id).data.metrics
    for key in ("test_accuracy", "test_precision", "test_recall", "test_f1"):
        if key in run_metrics:
            client.set_model_version_tag(MODEL_NAME, version, key, f"{run_metrics[key]:.4f}")

    client.set_registered_model_alias(MODEL_NAME, args.alias, version)
    print(f"{MODEL_NAME}@{args.alias} -> v{version}")


if __name__ == "__main__":
    main()