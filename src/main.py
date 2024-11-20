import spacy
import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
SPACY_LABELS = ["LOC", "MISC", "ORG", "PER"]

train_path = os.path.join(DATA_PATH, "NER-de-train.tsv")

df = pd.read_csv(
    train_path,
    sep="\t",
    names=["index", "token", "label", "_eval"],
    comment="#",
    quoting=3
)

print(df.head(5))

nlp = spacy.load("de_core_news_sm")
