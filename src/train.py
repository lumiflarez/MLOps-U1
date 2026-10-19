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
from sklearn.pipeline import Pipeline

params = yaml.safe_load(open("params.yaml"))["train"]

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("jailbreak-detection")

def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short=8", "HEAD"], text=True).strip()

def data_md5() -> str:
    pointer = yaml.safe_load(open("data/raw/jailbreak.csv.dvc"))
    return pointer["outs"][0]["md5"]

train = pd.read_csv("data/processed/train.csv")
x_train, y_train = train["prompt"], train["label"]

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=params["max_features"],
                              ngram_range=(1, params["ngram_max"]))),
    ("clf", LogisticRegression(C=params["C"], max_iter=1000,
                               random_state=params["random_seed"])),
])

run_name = f"lr-C{params['C']}-f{params['max_features']}-n{params['ngram_max']}"

with mlflow.start_run(run_name=run_name) as run:
    mlflow.log_params(params)
    mlflow.set_tags({"git_commit": git_commit(), "dvc_raw_md5": data_md5()})
    mlflow.log_input(mlflow.data.from_pandas(train, source="data/processed/train.csv",
                                             targets="label", name="train"), context="training")

    model.fit(x_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(x_train))
    mlflow.log_metric("train_accuracy", train_acc)

    mlflow.sklearn.log_model(model, name="model")

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    Path("models/run_id.txt").write_text(run.info.run_id + "\n")

    print(f"run_id={run.info.run_id}  train_accuracy={train_acc:.4f}")