import os
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

EXPERIMENT = "jailbreak-detection"
LEARNING_CURVE_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]

def git_commit() -> str:
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"

def data_md5() -> str:
    try:
        pointer = yaml.safe_load(open("data/raw/jailbreak.csv.dvc"))
        return pointer["outs"][0]["md5"]
    except Exception:
        return "untracked"

def build_model(params: dict) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=params["max_features"],
                                  ngram_range=(1, params["ngram_max"]))),
        ("clf", LogisticRegression(C=params["C"], max_iter=1000,
                                   random_state=params["random_seed"])),
    ])

def log_learning_curve(params: dict, x: pd.Series, y: pd.Series) -> None:
    x_tr, x_val, y_tr, y_val = train_test_split(
        x, y, test_size=0.2, random_state=params["random_seed"], stratify=y)

    for step, frac in enumerate(LEARNING_CURVE_FRACTIONS):
        n = int(len(x_tr) * frac)
        x_sub, y_sub = x_tr.iloc[:n], y_tr.iloc[:n]
        if y_sub.nunique() < 2:         
            continue
        m = build_model(params).fit(x_sub, y_sub)
        mlflow.log_metric("lc_train_size", n, step=step)
        mlflow.log_metric("lc_train_accuracy",
                          accuracy_score(y_sub, m.predict(x_sub)), step=step)
        mlflow.log_metric("lc_val_accuracy",
                          accuracy_score(y_val, m.predict(x_val)), step=step)

def main() -> None:
    params = yaml.safe_load(open("params.yaml"))["train"]

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT)
    mlflow.set_system_metrics_sampling_interval(1)
    mlflow.set_system_metrics_samples_before_logging(1)

    train = pd.read_csv("data/processed/train.csv")
    x, y = train["prompt"], train["label"]

    run_name = f"lr-C{params['C']}-f{params['max_features']}-n{params['ngram_max']}"

    with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
        mlflow.log_params(params)
        mlflow.set_tags({
            "git_commit": git_commit(),
            "dvc_raw_md5": data_md5(),
            "random_seed": str(params["random_seed"]),
            "model_family": "TfidfVectorizer+LogisticRegression",
        })
        mlflow.log_input(mlflow.data.from_pandas(
            train, source="data/processed/train.csv",
            targets="label", name="train"), context="training")

        log_learning_curve(params, x, y)

        model = build_model(params).fit(x, y)
        train_acc = accuracy_score(y, model.predict(x))

        info = mlflow.sklearn.log_model(model, name="model")
        mlflow.log_metric("train_accuracy", train_acc, model_id=info.model_id)

        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, "models/model.pkl")
        Path("models/run_id.txt").write_text(run.info.run_id + "\n")

        print(f"run_id={run.info.run_id} model_id={info.model_id} "
              f"train_accuracy={train_acc:.4f}")

if __name__ == "__main__":
    main()