from model import load_model
from preprocess import preprocess
from train import train
from evaluate import evaluate

from __param__ import SAMPLE, TRAIN

nlp = load_model()

data = {
    "train": preprocess("sample" if SAMPLE else "train", nlp),
    "dev": preprocess("dev", nlp),
    "test": preprocess("test", nlp)
}

if TRAIN:
    train(nlp, data["train"], data["dev"])

evaluate(nlp, data["test"])
