import json
import sys
from pathlib import Path

import yaml

gate = yaml.safe_load(Path("params.yaml").read_text())["gate"]
metrics = json.loads(Path("metrics.json").read_text())

value = metrics[gate["metric"]]
if value < gate["threshold"]:
    print(f"FAIL: {gate['metric']}={value:.4f} < {gate['threshold']}")
    sys.exit(1)
print(f"PASS: {gate['metric']}={value:.4f} >= {gate['threshold']}")