from spacy.language import Language
from spacy.lang.de import German
import de_core_news_sm
import os
from tqdm import tqdm

from __param__ import BLANK, INCLUDE_PART_DERIV

model_name = f"model-{
    "blank" if BLANK else "pretrained"}{
    "-part-deriv" if INCLUDE_PART_DERIV else ""}"


def load_model(out_path: str) -> Language:
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
        nlp = de_core_news_sm.load()

    if os.path.exists(os.path.join(out_path, model_name)):
        nlp.from_disk(os.path.join(out_path, model_name))

    return nlp


def save_model(nlp: Language, out_path: str):
    """
    Save the model to the disk.

    :param nlp: The model to save.
    :param out_path: The path to the output directory.
    """
    tqdm.write(f"Saving {model_name}…")
    os.makedirs(out_path, exist_ok=True)
    nlp.to_disk(os.path.join(out_path, model_name))
