import os
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import yaml

TOLERANCE = 0.005
METRICS = ["test_accuracy", "test_precision", "test_recall", "test_f1"]


def sh(*cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def output_of(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def as_number(text):
    try:
        return int(text)
    except ValueError:
        return float(text)


def main():
    old_id = sys.argv[1]
    start = time.time()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    client = mlflow.MlflowClient()
    old = client.get_run(old_id)
    commit = old.data.tags["git_commit"]
    print(f"Reproducing run {old_id}\n  commit: {commit}\n  params: {old.data.params}")

    if output_of("git", "status", "--porcelain"):
        sys.exit("You have uncommitted changes. Commit them first, then run again.")
    branch = output_of("git", "rev-parse", "--abbrev-ref", "HEAD")

    try:
        sh("git", "checkout", "--quiet", commit)

        sh("dvc", "checkout", "--force")
        md5 = yaml.safe_load(open("data/raw/jailbreak.csv.dvc"))["outs"][0]["md5"]
        print("raw data md5 matches the run:", md5 == old.data.tags["dvc_raw_md5"])

        params = yaml.safe_load(open("params.yaml"))
        params["train"] = {k: as_number(v) for k, v in old.data.params.items()}
        Path("params.yaml").write_text(yaml.safe_dump(params, sort_keys=False))

        sh("dvc", "repro", "--force")
        new_id = Path("models/run_id.txt").read_text().strip()
        client.set_tag(new_id, "reproduces", old_id)

        new = client.get_run(new_id)
        ok = True
        for m in METRICS:
            a, b = old.data.metrics[m], new.data.metrics[m]
            delta = abs(a - b)
            ok = ok and delta <= TOLERANCE
            print(f"  {m:15s} old={a:.4f}  new={b:.4f}  delta={delta:.4f}")
    finally:
        sh("git", "checkout", "--quiet", "--force", branch)
        sh("dvc", "checkout", "--force")

    seconds = time.time() - start
    print(f"{'PASS' if ok else 'FAIL'}: new run {new_id}, took {seconds:.1f}s (limit 120s)")


if __name__ == "__main__":
    main()