from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert "message" in body
    assert body["status"] == "experimental"


def test_prediction_endpoint():
    payload = {
        "engine_rpm": 745.67,
        "lub_oil_pressure": 3.162,
        "fuel_pressure": 6.2017,
        "coolant_pressure": 2.1669,
        "lub_oil_temp": 76.8174,
        "coolant_temp": 78.3467,
        "model_name": "logistic_regression",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert body["model"] == "logistic_regression"
    assert body["predicted_dataset_class"] in [0, 1]
    assert "class_probabilities" in body
    assert "warning" in body

    probabilities = body["class_probabilities"]

    assert set(probabilities.keys()) == {"0", "1"}

    assert abs(sum(probabilities.values()) - 1.0) < 0.01


def test_invalid_model_name():
    payload = {
        "engine_rpm": 745.67,
        "lub_oil_pressure": 3.162,
        "fuel_pressure": 6.2017,
        "coolant_pressure": 2.1669,
        "lub_oil_temp": 76.8174,
        "coolant_temp": 78.3467,
        "model_name": "unknown_model",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 400


def test_invalid_sensor_input():
    payload = {
        "engine_rpm": -10,
        "lub_oil_pressure": 3.162,
        "fuel_pressure": 6.2017,
        "coolant_pressure": 2.1669,
        "lub_oil_temp": 76.8174,
        "coolant_temp": 78.3467,
        "model_name": "logistic_regression",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422