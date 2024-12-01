import os
import pandas as pd
import pickle
from spacy.language import Language
from spacy.training import Example
from tqdm import tqdm
from typing import Literal

from .__param__ import INCLUDE_PART_DERIV, DEBUG, OUT

data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "res")

label_map = {
    "B-PER": "B-PER",
    "B-PERpart": "B-PERpart",
    "B-PERderiv": "B-PERderiv",
    "I-PER": "I-PER",
    "I-PERpart": "I-PERpart",
    "I-PERderiv": "I-PERderiv",

    "B-LOC": "B-LOC",
    "B-LOCpart": "B-LOCpart",
    "B-LOCderiv": "B-LOCderiv",
    "I-LOC": "I-LOC",
    "I-LOCpart": "I-LOCpart",
    "I-LOCderiv": "I-LOCderiv",

    "B-ORG": "B-ORG",
    "B-ORGpart": "B-ORGpart",
    "B-ORGderiv": "B-ORGderiv",
    "I-ORG": "I-ORG",
    "I-ORGpart": "I-ORGpart",
    "I-ORGderiv": "I-ORGderiv",

    "B-OTH": "B-MISC",
    "B-OTHpart": "B-MISCpart",
    "B-OTHderiv": "B-MISCderiv",
    "I-OTH": "I-MISC",
    "I-OTHpart": "I-MISCpart",
    "I-OTHderiv": "I-MISCderiv",

    "O": "O"
} if INCLUDE_PART_DERIV else {
    "B-PER": "B-PER",
    "I-PER": "I-PER",
    "B-PERpart": "B-PER",
    "I-PERpart": "I-PER",
    "B-PERderiv": "B-PER",
    "I-PERderiv": "I-PER",

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

    "O": "O"
}


def preprocess(dataset: Literal["train", "dev", "sample"], nlp: Language) -> list[Example]:
    """
    Load the dataset and preprocess it for the model.

    :param dataset: The dataset to load
    :return: A tuple containing the examples and the labels
    """
    cache_path = os.path.join(OUT,
                              f"{dataset}{'-partderiv' if INCLUDE_PART_DERIV else ''}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as cache_file:
            tqdm.write(f"Loading cached {dataset} examples from {cache_path}…")
            return pickle.load(cache_file)

    df = pd.read_csv(
        os.path.join(data_path, f"NER-de-{dataset}.tsv"),
        sep="\t",
        names=["index", "token", "label", "_eval"],
        comment="#",
        quoting=3,
        na_filter=True,
        skip_blank_lines=False
    )

    if DEBUG:
        print("\n Data set details:")
        print(df.head())
        print("Size: ", len(df))
        print("List of all Labels:\n", str(df["label"].unique()))

    df["label"] = df["label"].replace(label_map)  # normalize the labels
    ner = nlp.get_pipe("ner")
    for label in df["label"].unique():
        ner.add_label(str(label))
        if DEBUG:
            print("Added Label: ", label)

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

    os.makedirs(OUT, exist_ok=True)
    with open(cache_path, "wb") as cache_file:
        pickle.dump(examples, cache_file)
        if DEBUG:
            print(f"Cached {len(examples)} examples to {cache_path}")

    return examples


def unprocess(label: str) -> str:
    """
    Unprocess the label.

    :param label: The label to unprocess.
    :return: The unprocessed label.
    """
    assert label in label_map.values(), f"Unknown label: {label}"
    reversed_label_map = {v: k for k, v in label_map.items()}
    return reversed_label_map[label]
