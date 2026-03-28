import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


def load_data():
    return load_iris()


def train_model():
    data = load_data()
    model = RandomForestClassifier()
    model.fit(data.data, data.target)
    return model


def save_model(model):
    joblib.dump(model, "model.joblib")


if __name__ == "__main__":
    model = train_model()
    save_model(model)
    print("Model saved!")
