from spacy.language import Language
from spacy.lang.de import German
from spacy.cli import download
from spacy import load
import os
from tqdm import tqdm

from __param__ import BLANK, INCLUDE_PART_DERIV, OUT

model_name = \
    f"model-{'blank' if BLANK else 'pretrained'}{'-part-deriv' if INCLUDE_PART_DERIV else ''}"


def load_model() -> Language:
    """
    Load the model from the disk if it exists, otherwise create a new model.

    :param out_path: The path to the output directory.
    :return: The loaded model.
    """
    tqdm.write(f"Loading {model_name}…")
    if BLANK:
        nlp = German()
        nlp.add_pipe("ner")
    else:
        download("de_core_news_sm")
        nlp = load("de_core_news_sm")

    if os.path.exists(os.path.join(OUT, model_name)):
        nlp.from_disk(os.path.join(OUT, model_name))

    return nlp


def save_model(nlp: Language):
    """
    Save the model to the disk.

    :param nlp: The model to save.
    :param out_path: The path to the output directory.
    """
    tqdm.write(f"Saving {model_name}…")
    os.makedirs(OUT, exist_ok=True)
    nlp.to_disk(os.path.join(OUT, model_name))
