import os
from spacy.language import Language
from spacy.lang.de import German
from spacy.cli import download
from spacy import load
from tqdm import tqdm

from .__param__ import BLANK, INCLUDE_PART_DERIV, MODEL_PATH, DEBUG

model_name = \
    f"model-{'blank' if BLANK else 'pretrained'}{'-partderiv' if INCLUDE_PART_DERIV else ''}"
model_path = os.path.join(MODEL_PATH, model_name)


def load_model() -> Language:
    """
    Load the model from the disk if it exists, otherwise create a new model.

    :return: The loaded model.
    """
    if BLANK:
        nlp = German()
        nlp.add_pipe("ner")
    else:
        try:
            nlp = load("de_core_news_sm")
        except OSError:
            download("de_core_news_sm")
            nlp = load("de_core_news_sm")

    if os.path.exists(model_path):
        tqdm.write(f"Loading model from {model_path}…")
        nlp.from_disk(model_path)
    else:
        tqdm.write(f"Created new {model_name}.")

    return nlp


def save_model(nlp: Language):
    """
    Save the model to the disk.

    :param nlp: The model to save.
    """
    if DEBUG:
        print(f"\tSaving model to {model_path}…")
    os.makedirs(model_path, exist_ok=True)
    nlp.to_disk(model_path)
