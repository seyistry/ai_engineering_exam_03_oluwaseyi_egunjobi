from typing import List, Union

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field


# Expected feature names for the wine dataset (common set)
FEATURE_NAMES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def load_artifact(path: str):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None
    except Exception as e:
        # return None and let endpoints raise a clear error
        return None


model = load_artifact("model.pkl")
scaler = load_artifact("scaler.pkl")


class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


class MultipleInputs(BaseModel):
    inputs: List[WineFeatures]


app = FastAPI(
    title="Wine Quality Predictor",
    description="Predict wine quality class (Best, Good, Average, Bad) from chemical properties.",
    version="1.0",
)


def map_quality_to_label(q: int) -> str:
    """Map a numeric quality score to a human-friendly class label.

    Assumption: model returns integer-like quality (e.g., 3-9). We map ranges to labels.
    - 7 and above: Best
    - 6: Good
    - 5: Average
    - 4 and below: Bad
    """
    try:
        q = int(np.round(q))
    except Exception:
        return "Unknown"
    if q >= 7:
        return "Best"
    if q == 6:
        return "Good"
    if q == 5:
        return "Average"
    return "Bad"


@app.get("/", tags=["health"])
def welcome():
    return {"message": "Welcome to the Wine Quality Predictor API. See /docs for Swagger UI."}


@app.post("/predict", tags=["prediction"])
def predict(
        payload: Union[WineFeatures, MultipleInputs],
):
    # Check artifacts
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact 'model.pkl' not found or failed to load.",
        )

    # normalize payload to list
    if isinstance(payload, MultipleInputs):
        items = payload.inputs
    elif isinstance(payload, WineFeatures):
        items = [payload]
    else:
        # Should not happen due to pydantic, but keep safe
        raise HTTPException(status_code=400, detail="Invalid input format")

    # Build DataFrame
    try:
        df = pd.DataFrame([item.dict()
                          for item in items], columns=FEATURE_NAMES)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to construct features: {e}")

    # Apply scaler if available
    try:
        X = df.values.astype(float)
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed during scaling: {e}")

    # Predict
    try:
        preds = model.predict(X_scaled)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Model prediction failed: {e}")

    # Map predictions
    results = []
    for p in preds:
        try:
            label = map_quality_to_label(p)
            results.append(
                {"quality": int(np.round(float(p))), "label": label})
        except Exception:
            results.append({"quality": None, "label": "Unknown"})

    return {"status": "success", "predictions": results}


if __name__ == "__main__":
    # When running directly, start uvicorn for local testing
    uvicorn.run(app, host="0.0.0.0", port=8000)
