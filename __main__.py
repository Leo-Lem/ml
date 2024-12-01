from src import load_model, preprocess, train, evaluate, SAMPLE, TRAIN

nlp = load_model()

data = {
    "train": preprocess("sample" if SAMPLE else "train", nlp),
    "dev": preprocess("dev", nlp),
    "test": preprocess("test", nlp)
}

if TRAIN:
    train(nlp, data["train"], data["dev"])

evaluate(nlp, data["test"])
