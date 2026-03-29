from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_predict_positive():
    response = client.post("/predict", json={"text": "I love this, it is great!"})
    assert response.status_code == 200
    assert response.json()["prediction"] in ["positive", "neutral", "negative"]


def test_predict_negative():
    response = client.post("/predict", json={"text": "This is terrible and awful"})
    assert response.status_code == 200
    assert response.json()["prediction"] in ["positive", "neutral", "negative"]


def test_predict_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_missing_field():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_invalid_input():
    response = client.post("/predict", json={"text": 123})
    assert response.status_code == 422
