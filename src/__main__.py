from model import load_model
from preprocess import preprocess
from train import train
from evaluate import evaluate

nlp = load_model()

data = {
    "train": preprocess("train", nlp),
    "dev": preprocess("dev", nlp),
    "test": preprocess("sample", nlp)
}

train(nlp, data["train"], data["dev"])

evaluate(nlp, data["test"])
