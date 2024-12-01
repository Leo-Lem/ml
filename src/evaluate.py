import os
from tqdm import tqdm
from spacy.language import Language
from spacy.training import Example
import subprocess

from .__param__ import OUT, DEBUG
from .preprocess import unprocess


def evaluate(nlp: Language, data: list[Example]):
    """
    Evaluate the NER model.

    :param nlp: The NLP object.
    :param data: The evaluation data.
    """
    predictions = os.path.join(OUT, "predictions.tsv")
    os.makedirs(OUT, exist_ok=True)

    with open(predictions, "w", encoding="utf-8") as f:
        for example in tqdm(data, desc="Predicting", unit="sentence"):
            gold_doc, predicted_doc = example.reference, nlp(example.text)

            for token in gold_doc:
                gold_label = "O"
                for ent in gold_doc.ents:
                    if token.idx == ent.start_char:
                        gold_label = f"B-{unprocess(ent.label_)}"
                        break
                    elif ent.start_char < token.idx and token.idx < ent.end_char:
                        gold_label = f"I-{unprocess(ent.label_)}"
                        break

                predicted_label = "O"
                for ent in predicted_doc.ents:
                    if token.idx == ent.start_char:
                        predicted_label = f"B-{unprocess(ent.label_)}"
                        break
                    elif ent.start_char < token.idx and token.idx < ent.end_char:
                        predicted_label = f"I-{unprocess(ent.label_)}"
                        break

                f.write(
                    f"{token.i + 1}\t{token.text}\t" +
                    f"{gold_label}\tO\t" +
                    f"{predicted_label}\tO\n"
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
