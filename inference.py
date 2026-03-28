import joblib
from sklearn.datasets import load_iris


def load_model():
    return joblib.load("model.joblib")


def predict(features: dict) -> str:
    model = load_model()
    iris = load_iris()
    data = [
        [
            features["sepal_length"],
            features["sepal_width"],
            features["petal_length"],
            features["petal_width"],
        ]
    ]
    prediction = model.predict(data)
    return iris.target_names[prediction[0]]
