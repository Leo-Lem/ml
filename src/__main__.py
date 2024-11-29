import os

from model import load_model, save_model
from preprocess import preprocess
from train import train
from evaluate import evaluate

out_path = os.path.join(os.path.dirname(__file__), "..", ".out")


nlp = load_model(out_path)
data = {
    "train": preprocess("train", nlp),
    "dev": preprocess("dev", nlp),
    "test": preprocess("sample", nlp)
}

train(nlp, data["train"], data["dev"])

save_model(nlp, out_path)

evaluate(nlp, data["test"], out_path)
