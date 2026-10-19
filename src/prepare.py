from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["prepare"]

df = pd.read_csv("data/raw/jailbreak.csv")

df = df.dropna(subset=["prompt", "type"])
df = df.drop_duplicates(subset="prompt")

df["label"] = df["type"].map({"benign": 0, "jailbreak": 1})
df = df[["prompt", "label"]]

train, test = train_test_split(
    df,
    test_size=params["test_size"],
    random_state=params["random_seed"],
    stratify=df["label"],
)

Path("data/processed").mkdir(parents=True, exist_ok=True)
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print(f"train: {len(train)} rows, test: {len(test)} rows")