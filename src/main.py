import spacy
from spacy.training import Example
import pandas as pd
import os
import subprocess  # for running eval script

paths = {
    "data": os.path.join(os.path.dirname(__file__), "..", "data"),
    "model": os.path.join(os.path.dirname(__file__), "..", "model"),
    "eval": os.path.join(os.path.dirname(__file__), "..", "eval"),
}
datasets = {
    "train": os.path.join(paths["data"], "NER-de-train.tsv"),
    "dev": os.path.join(paths["data"], "NER-de-dev.tsv"),
    "sample": os.path.join(paths["data"], "NER-de-sample.tsv"),
}
perl_script_path = os.path.join(paths["eval"], "nereval.perl")
predictions_path = os.path.join(paths["eval"], "predictions.tsv")

LABELS = {
    "PER": "PER",
    "LOC": "LOC",
    "ORG": "ORG",
    "MISC": "OTH",
}


print("\n<=== TRAIN ===>")
print("Loading the model…")
nlp = spacy.load("de_core_news_sm")
nlp.from_disk(paths["model"])
ner = nlp.get_pipe("ner")


print("Loading the data…")
train_df = pd.read_csv(
    datasets["train"],
    sep="\t",
    names=["index", "token", "label", "_eval"],
    comment="#",
    quoting=3,
    na_filter=False
)
dev_df = pd.read_csv(
    datasets["dev"],
    sep="\t",
    names=["index", "token", "label", "_eval"],
    comment="#",
    quoting=3,
    na_filter=False
)

for label, replacement in LABELS.items():  # TODO: ?:part|deriv)?
    train_df["label"] = train_df["label"]\
        .astype(str).replace(replacement, label)
    dev_df["label"] = dev_df["label"]\
        .astype(str).replace(replacement, label)

for label in train_df["label"].unique():
    ner.add_label(label)

print("Converting to spacy format…")


def convert_to_spacy_format(df):
    sentences = []
    current_sentence = []
    entities = []
    current_entity = None
    offset = 0

    # Fill missing or misinterpreted tokens with a default value (e.g., "UNKNOWN")
    df["token"] = df["token"].fillna("UNKNOWN").astype(str)

    for i, row in df.iterrows():
        if pd.isna(row["index"]):  # End of a sentence
            if current_sentence:
                sentences.append((current_sentence, {"entities": entities}))
            current_sentence = []
            entities = []
            current_entity = None
            offset = 0
            continue

        token = str(row["token"])  # Ensure token is a string
        label = row["label"]
        current_sentence.append(token)

        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = [offset, offset + len(token), label[2:]]
        elif label.startswith("I-") and current_entity:
            current_entity[1] = offset + len(token)
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

        offset += len(token) + 1  # Include space

    if current_sentence:
        sentences.append((current_sentence, {"entities": entities}))

    return sentences


train_data = convert_to_spacy_format(train_df)
dev_data = convert_to_spacy_format(dev_df)

# Convert sentences and entities into spaCy training Examples
examples = []
for text, annotations in train_data:
    doc = nlp.make_doc(" ".join(text))
    example = Example.from_dict(doc, annotations)
    examples.append(example)

print(f"Training on {len(examples)} examples...")

# Start the training process
optimizer = nlp.initialize()
for i in range(20):  # Adjust the number of iterations as needed
    losses = {}
    nlp.update(examples, sgd=optimizer, losses=losses)
    print(f"Iteration {i + 1}, Losses: {losses}")

print("Model training complete!")
...

print("Saving the model…")
nlp.to_disk(paths["model"])


print("\n<=== TEST ===>")
doc = nlp("Die Deutsche Flugsicherung (DFS) beschloss ein Flugverbot für alle internationalen Flughäfen mit Ausnahme der beiden Berliner Flughäfen bis 2.00 Uhr nachts.")
# TODO: test using test dataset

print("\nWriting the predictions into TSV file...")
with open(predictions_path, "w", encoding="utf-8") as output_file:
    for token in doc:
        pred_label = "O"  # Default label
        for ent in doc.ents:
            if token.idx >= ent.start_char and token.idx < ent.end_char:
                # temporary normalization. we only need this while using the spacy model. once we replace the pre-trained model with our own, this normalization can be deleted
                pred_label = LABELS.get(ent.label_, ent.label_)
                break

        output_file.write(
            f"{token.i + 1}\t{token.text}\t{pred_label}\tO\tO\tO\n")
print(f"Predictions written to {predictions_path}")

print("\nRunning the evaluation script…")
with open(predictions_path, "rb") as file:
    eval = subprocess.run(
        ["perl", perl_script_path],
        input=file.read(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
print(eval.stdout.decode("utf-8"))
