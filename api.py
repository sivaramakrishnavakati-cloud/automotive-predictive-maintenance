from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

FEATURES = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
]

MODEL_FILES = {
    "logistic_regression": "logistic_regression.joblib",
    "random_forest": "random_forest.joblib",
    "gradient_boosting": "gradient_boosting.joblib",
}

app = FastAPI(
    title="AI Engine Condition Prediction API",
    description="Experimental engine-condition classification API",
    version="1.0.0",
)


class SensorInput(BaseModel):
    engine_rpm: float = Field(gt=0, le=10000)
    lub_oil_pressure: float = Field(ge=0, le=1000)
    fuel_pressure: float = Field(ge=0, le=1000)
    coolant_pressure: float = Field(ge=0, le=1000)
    lub_oil_temp: float = Field(ge=-100, le=500)
    coolant_temp: float = Field(ge=-100, le=500)
    model_name: str = "logistic_regression"


def load_model(model_name: str):
    if model_name not in MODEL_FILES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported model name.",
        )

    model_path = MODEL_DIR / MODEL_FILES[model_name]

    if not model_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found: {model_path.name}",
        )

    return joblib.load(model_path)


@app.get("/")
def home():
    return {
        "message": "Engine Condition Prediction API is running",
        "status": "experimental",
    }


@app.post("/predict")
def predict(data: SensorInput):
    model = load_model(data.model_name)

    input_data = pd.DataFrame(
        [[
            data.engine_rpm,
            data.lub_oil_pressure,
            data.fuel_pressure,
            data.coolant_pressure,
            data.lub_oil_temp,
            data.coolant_temp,
        ]],
        columns=FEATURES,
    )

    prediction = int(model.predict(input_data)[0])

    probabilities = {}

    if hasattr(model, "predict_proba"):
        values = model.predict_proba(input_data)[0]

        probabilities = {
            str(label): round(float(probability), 4)
            for label, probability in zip(model.classes_, values)
        }

    return {
        "model": data.model_name,
        "predicted_dataset_class": prediction,
        "class_probabilities": probabilities,
        "warning": (
            "Experimental dataset classification only. "
            "This is not a confirmed vehicle diagnosis."
        ),
    }