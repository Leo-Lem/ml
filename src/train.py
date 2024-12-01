import pandas as pd
from random import shuffle
from spacy.util import minibatch
from spacy.language import Language
from spacy.training import Example
from tqdm import tqdm, trange

from model import save_model
from __param__ import EPOCHS, BATCH_SIZE, STOP_EARLY_AFTER, DEBUG


def train(nlp: Language, data: list[Example], eval_data: list[Example]) -> pd.DataFrame:
    """
    Train the NER model.

    :param nlp: The NLP object.
    :param data: The training data.
    :param eval_data: The evaluation data.
    """
    try:
        optimizer = nlp.resume_training()
        nlp.update(data[0:1], sgd=optimizer)  # check if training is possible
        if DEBUG:
            print("Resuming training…")
    except KeyError:
        optimizer = nlp.begin_training()
        if DEBUG:
            print("Starting training…")

    scores = pd.DataFrame(columns=["ents_f", "ents_p", "ents_r"])
    for epoch in trange(EPOCHS, desc="Training", unit="epoch"):
        shuffle(data)
        batches = minibatch(data, size=BATCH_SIZE)
        for batch in tqdm(batches, total=len(data) // BATCH_SIZE, unit="batch", desc=f"Epoch {epoch + 1}", leave=False):
            nlp.update(batch, sgd=optimizer)

        scores[epoch] = nlp.evaluate(eval_data)
        tqdm.write(
            f"\tEpoch {epoch + 1} | F1-score: {scores[epoch]['ents_f']:.4f} | Precision: {scores[epoch]['ents_p']:.4f} | Recall: {scores[epoch]['ents_r']:.4f}")

        if STOP_EARLY_AFTER > 0 and epoch >= STOP_EARLY_AFTER and all(scores[epoch]["ents_f"] <= scores[epoch - i - 1]["ents_f"] for i in range(STOP_EARLY_AFTER)):
            tqdm.write(
                f"\tModel has not improved for {STOP_EARLY_AFTER} epochs, stopping…")
            break

        save_model(nlp)
    return scores
