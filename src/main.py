import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
train_path = os.path.join(DATA_PATH, "NER-de-train.tsv")

df = pd.read_csv(
    train_path,
    sep="\t",
    names=["index", "token", "label", "_eval"],
    quoting=3
)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.head(5))
