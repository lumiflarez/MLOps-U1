import argparse
import os

import mlflow
import optuna
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, cross_val_score
import mlflow.sklearn
from sklearn.metrics import accuracy_score

from train import EXPERIMENT, build_model, data_md5, git_commit, log_learning_curve

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--cv-folds", type=int, default=3)
    args = parser.parse_args()

    seed = yaml.safe_load(open("params.yaml"))["train"]["random_seed"]

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT)
    mlflow.set_system_metrics_sampling_interval(1)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train = pd.read_csv("data/processed/train.csv")
    x, y = train["prompt"], train["label"]
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "C": trial.suggest_float("C", 0.01, 1000.0, log=True),
            "max_features": trial.suggest_categorical("max_features", [2000, 5000, 10000, 20000, 50000]),
            "ngram_max": trial.suggest_int("ngram_max", 1, 3),
            "random_seed": seed,
        }
        with mlflow.start_run(run_name=f"trial-{trial.number:02d}", nested=True):
            mlflow.log_params(params)
            mlflow.set_tag("optuna_trial", str(trial.number))

            scores = cross_val_score(build_model(params), x, y, cv=cv, scoring="accuracy")
            for fold, score in enumerate(scores):
                mlflow.log_metric("cv_fold_accuracy", score, step=fold)
            mlflow.log_metric("cv_accuracy_mean", scores.mean())
            mlflow.log_metric("cv_accuracy_std", scores.std())

        return scores.mean() 

    with mlflow.start_run(run_name="optuna-hpo", log_system_metrics=True):
        mlflow.set_tags({
            "git_commit": git_commit(),
            "dvc_raw_md5": data_md5(),
            "random_seed": str(seed),
            "stage": "hpo",
        })
        mlflow.log_params({
            "n_trials": args.n_trials,
            "cv_folds": args.cv_folds,
            "sampler": "TPE",
            "space_C": "0.01-1000 log",
            "space_max_features": "2000,5000,10000,20000,50000",
            "space_ngram_max": "1-3",
        })
        mlflow.log_input(mlflow.data.from_pandas(
            train, source="data/processed/train.csv",
            targets="label", name="train"), context="training")

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=args.n_trials)

        best_so_far = 0.0
        for t in study.trials:
            best_so_far = max(best_so_far, t.value)
            mlflow.log_metric("trial_cv_accuracy", t.value, step=t.number)
            mlflow.log_metric("best_cv_accuracy_so_far", best_so_far, step=t.number)

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_cv_accuracy", study.best_value)
        mlflow.log_dict(study.best_params, "best_params.json")
        mlflow.log_text(study.trials_dataframe().to_csv(index=False), "trials.csv")

        best = {**study.best_params, "random_seed": seed}
        log_learning_curve(best, x, y)
        final = build_model(best).fit(x, y)
        info = mlflow.sklearn.log_model(final, name="model")
        mlflow.log_metric("best_train_accuracy",
                          accuracy_score(y, final.predict(x)), model_id=info.model_id)

        print(f"Geriausias bandymas #{study.best_trial.number}: "
              f"cv_accuracy={study.best_value:.4f} params={study.best_params}")

if __name__ == "__main__":
    main()