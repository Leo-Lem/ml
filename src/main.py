import random
from spacy.util import minibatch
from spacy.training import Example
import spacy
import pandas as pd
import os

base_path = os.path.join(os.path.dirname(__file__), "..")
data_path = os.path.join(base_path, "data")
model_path = os.path.join(base_path, "model")
paths = {
    "train": os.path.join(data_path, "NER-de-train.tsv"),
    "dev": os.path.join(data_path, "NER-de-dev.tsv"),
    "sample": os.path.join(data_path, "NER-de-sample.tsv"),
}

LABELS = {
    "PER": "PER",
    "LOC": "LOC",
    "ORG": "ORG",
    "MISC": "OTH",
}


print("Loading the model…")
nlp = spacy.load("de_core_news_sm")
nlp.from_disk(model_path)
ner = nlp.get_pipe("ner")


print("Preparing the data…")
df = pd.read_csv(
    paths["sample"],
    sep="\t",
    names=["index", "token", "label", "_eval"],
    comment="#",
    quoting=3
)

# TODO: ?:part|deriv)?
for label, replacement in LABELS.items():
    df["label"] = df["label"].replace(replacement, label)

for label in df["label"].unique():
    ner.add_label(label)


print("Training the model…")
# TODO: train using train and dev dataset
nlp.to_disk(model_path)


print("Testing the model…")
doc = nlp("Die Deutsche Flugsicherung (DFS) beschloss ein Flugverbot für alle internationalen Flughäfen mit Ausnahme der beiden Berliner Flughäfen bis 2.00 Uhr nachts.")
for ent in doc.ents:
    print(ent.text, ent.label_)
# TODO: test using test dataset
