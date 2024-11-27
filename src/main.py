import spacy
from spacy.training import Example
from spacy.util import minibatch
import pandas as pd
import os
import subprocess  # for running eval script
from tqdm import tqdm, trange
from random import shuffle

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


def load_for_spacy(dataset: str) -> list[Example]:
    df = pd.read_csv(
        datasets[dataset],
        sep="\t",
        names=["index", "token", "label", "_eval"],
        comment="#",
        quoting=3,
        na_filter=True,
        skip_blank_lines=False
    )

    # TODO: ?:part|deriv)?
    # replace the labels with the ones used in the model
    df["label"] = df["label"].replace(LABELS)

    for label in df["label"].unique():
        ner.add_label(str(label))

    examples = []
    current_sentence, entities, current_entity, offset = [], [], None, 0
    for _, row in tqdm(df.iterrows(), f"Loading {dataset} sentences", total=len(df), unit="rows"):
        if pd.isna(row["index"]):
            if current_sentence:
                examples.append(Example.from_dict(
                    nlp.make_doc(" ".join(current_sentence)), {"entities": entities}))
            current_sentence = []
            entities = []
            current_entity = None
            offset = 0
            continue
        token = str(row["token"])  # Ensure token is a string
        label = str(row["label"])  # Ensure label is a string

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

        offset += len(token) + 1  # Include space after token

    return examples


train_data = load_for_spacy("sample")
dev_data = load_for_spacy("dev")


def train(batch_size: int, optimizer: str):
    for epoch in trange(10, desc="Training", unit="epoch"):
        losses = {}
        shuffle(train_data)  # Shuffle data each epoch
        batches = minibatch(train_data, size=batch_size)  # Train in batches

        for batch in tqdm(batches, total=len(train_data) // batch_size, unit="batch", leave=False):
            nlp.update(batch, sgd=optimizer, losses=losses)

        scorer = nlp.evaluate(dev_data)
        print(
            f"Epoch {epoch + 1} | F1-score: {scorer['ents_f']:.4f} | Precision: {scorer['ents_p']:.4f} | Recall: {scorer['ents_r']:.4f}")


train(1024, nlp.initialize())

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
