import random
from spacy.util import minibatch
from spacy.training import Example
import spacy
import pandas as pd
import os
import subprocess # for running eval script

base_path = os.path.join(os.path.dirname(__file__), "..")
data_path = os.path.join(base_path, "data")
model_path = os.path.join(base_path, "model")
eval_path = os.path.join(base_path, "eval")
paths = {
    "train": os.path.join(data_path, "NER-de-train.tsv"),
    "dev": os.path.join(data_path, "NER-de-dev.tsv"),
    "sample": os.path.join(data_path, "NER-de-sample.tsv"),
}
perl_script_path = os.path.join(eval_path, "nereval.perl")
predictions_path = os.path.join(eval_path, "predictions.tsv")

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

print("Writing the predictions into TSV file...")



with open(predictions_path, "w", encoding="utf-8") as output_file:
    for token in doc:
        pred_label = "O"  # Default label
        for ent in doc.ents:
            if token.idx >= ent.start_char and token.idx < ent.end_char:
                pred_label = LABELS.get(ent.label_, ent.label_)  # temporary normalization. we only need this while using the spacy model. once we replace the pre-trained model with our own, this normalization can be deleted
                break

        output_file.write(f"{token.i + 1}\t{token.text}\t{pred_label}\tO\tO\tO\n")

print(f"Predictions written to {predictions_path}")

print("Evaluating the model performance...")

with open(predictions_path, "rb") as file:
    eval = subprocess.run(
        ["perl", perl_script_path],
        input=file.read(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
print("<=== EVAL ===>")
print(eval.stdout.decode("utf-8"))