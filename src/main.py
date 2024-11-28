import spacy
from spacy.training import Example
from spacy.util import minibatch
import pandas as pd
import os
import subprocess  # for running eval script
from tqdm import tqdm, trange
from random import shuffle

#training parameters
no_epochs = 7

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

# Label normalization map for GermaNER -> SpaCy (general)
label_mapping_spacy = {
    "B-PER": "B-PERSON",
    "I-PER": "I-PERSON",
    "B-LOC": "B-LOC",
    "I-LOC": "I-LOC",
    "B-ORG": "B-ORG",
    "I-ORG": "I-ORG",
    "B-OTH": "B-MISC",
    "I-OTH": "I-MISC",

}

# Label normalization map for GermaNER -> SpaCy (including partial tag normalization)
label_mapping_spacy_full = {
    "B-PER": "B-PERSON",
    "I-PER": "I-PERSON",
    "B-PERpart": "B-PERSON",
    "I-PERpart": "I-PERSON",
    "B-PERderiv": "B-PERSON",
    "I-PERderiv": "I-PERSON",

    "B-LOC": "B-LOC",
    "I-LOC": "I-LOC",
    "B-LOCpart": "B-LOC",
    "I-LOCpart": "I-LOC",
    "B-LOCderiv": "B-LOC",
    "I-LOCderiv": "I-LOC",

    "B-ORG": "B-ORG",
    "I-ORG": "I-ORG",
    "B-ORGpart": "B-ORG",
    "I-ORGpart": "I-ORG",
    "B-ORGderiv": "B-ORG",
    "I-ORGderiv": "I-ORG",

    "B-OTH": "B-MISC",
    "I-OTH": "I-MISC",
    "B-OTHpart": "B-MISC",
    "I-OTHpart": "I-MISC",
    "B-OTHderiv": "B-MISC",
    "I-OTHderiv": "I-MISC",

    "O": "O",  # Outside tokens
}

# Choose label map (parameter)
LABELS = label_mapping_spacy_full  # pick either label_mapping_spacy OR label_mapping_spacy_full

print("\n<=== TRAIN ===>")
print("Loading the model…")
nlp = spacy.load("de_core_news_sm")
nlp.from_disk(paths["model"])

if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True) # initialize
else:
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

    # debugging
    #df = df.head(8000)
    print("\n Data set details:")
    print(df.head())
    print("Size: ",len(df))
    print("List of all Labels:\n", str(df["label"].unique()))
    # debugging end


    # replace the labels with the ones used in the model
    df["label"] = df["label"].replace(LABELS)

    for label in df["label"].unique():
        ner.add_label(str(label))
        print("Added label: ", str(label)) # for debug

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


train_data = load_for_spacy("train")
dev_data = load_for_spacy("dev")


def train(batch_size: int, optimizer: str):
    for epoch in trange(no_epochs, desc="Training", unit="epoch"):
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
                ent_prefix = "B-" if token.idx == ent.start_char else "I-" #eval needs "BIO-tags"
                pred_label = ent_prefix + LABELS.get(ent.label_, ent.label_)
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
