import pandas as pd
from datasets import load_dataset

ds = load_dataset("jackhhao/jailbreak-classification")
df = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas()], ignore_index=True)
df.to_csv("data/raw/jailbreak.csv", index=False)
