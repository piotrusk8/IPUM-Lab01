import joblib
from sentence_transformers import SentenceTransformer

CLASSES = {0: "negative", 1: "neutral", 2: "positive"}


def load_transformer():
    return SentenceTransformer("sentence_transformer.model")


def load_classifier():
    return joblib.load("classifier.joblib")


def predict(text: str) -> str:
    transformer = load_transformer()
    classifier = load_classifier()
    embedding = transformer.encode([text])
    prediction = classifier.predict(embedding)
    return CLASSES[prediction[0]]
