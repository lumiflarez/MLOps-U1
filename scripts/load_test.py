import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--output", default="docs/load-test.json")
    args = parser.parse_args()

    prompts = (pd.read_csv("data/processed/test.csv")["prompt"]
               .sample(n=args.n, replace=True, random_state=42).tolist())

    latencies, errors = [], 0
    t0 = time.perf_counter()
    for prompt in prompts:
        start = time.perf_counter()
        r = requests.post(f"{args.url}/predict", json={"prompt": prompt}, timeout=10)
        latencies.append((time.perf_counter() - start) * 1000)
        if r.status_code != 200:
            errors += 1
    total_s = time.perf_counter() - t0

    report = {
        "requests": args.n,
        "errors": errors,
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "throughput_rps": round(args.n / total_s, 1),
    }
    print(json.dumps(report, indent=2))
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()