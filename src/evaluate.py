import os
from tqdm import tqdm
from spacy.language import Language
from spacy.training import Example
import subprocess

from __param__ import OUT


def evaluate(nlp: Language, data: list[Example]):
    """
    Evaluate the NER model.

    :param nlp: The NLP object.
    :param data: The evaluation data.
    """

    predictions = os.path.join(OUT, "predictions.tsv")
    os.makedirs(OUT, exist_ok=True)

    with open(predictions, "w", encoding="utf-8") as f:
        for example in tqdm(data, desc="Predicting…"):
            gold_doc = example.reference
            predicted_doc = nlp(example.text)

            for token in gold_doc:
                gold_label = "O"
                for ent in gold_doc.ents:
                    if token.idx >= ent.start_char and token.idx < ent.end_char:
                        ent_prefix = "B-" if token.idx == ent.start_char else "I-"
                        gold_label = ent_prefix + ent.label_
                        break

                predicted_label = "O"
                for ent in predicted_doc.ents:
                    if token.idx >= ent.start_char and token.idx < ent.end_char:
                        ent_prefix = "B-" if token.idx == ent.start_char else "I-"
                        predicted_label = ent_prefix + ent.label_
                        break

                f.write(
                    f"{token.i + 1}\t{token.text}" +
                    f"\t{gold_label}\t{gold_label}" +
                    f"\t{predicted_label}\t{predicted_label}\n"
                )
            f.write("\n")  # Add a blank line between sentences

    tqdm.write("Running the evaluation script…")
    with open(predictions, "rb") as f:
        eval = subprocess.run(
            ["perl", os.path.join(os.path.dirname(__file__), "nereval.perl")],
            input=f.read(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        tqdm.write(eval.stdout.decode("utf-8"))
