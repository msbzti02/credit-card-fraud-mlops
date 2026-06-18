import os
import sys
import warnings
from datetime import datetime
from typing import List

import mlflow
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import load_config

warnings.filterwarnings("ignore")

config = load_config()
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

MODEL_NAME = config["mlflow"]["registry_model_name"]

print(f"[INFO] Loading model '{MODEL_NAME}' from Production stage ...")
try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
    model_loaded = True
    print("[INFO] Model loaded successfully")
except Exception as e:
    print(f"[WARN] Could not load model: {e}")
    print("[WARN] Server will start but predictions will fail.")
    model = None
    model_loaded = False

try:
    _X = pd.read_csv(
        os.path.join(config["data"]["processed_dir"], "X_test.csv"), nrows=1
    )
    FEATURE_NAMES = list(_X.columns)
    N_FEATURES = len(FEATURE_NAMES)
except Exception:
    FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]
    N_FEATURES = 30
print(f"[INFO] Expecting {N_FEATURES} features per transaction")


class Transaction(BaseModel):

    features: List[float] = Field(
        ...,
        min_length=N_FEATURES,
        max_length=N_FEATURES,
        description=f"{N_FEATURES} numerical features",
    )


class BatchTransactions(BaseModel):

    transactions: List[Transaction]


class PredictionResponse(BaseModel):

    prediction: int = Field(description="0 = Legitimate, 1 = Fraud")
    probability: float = Field(description="Fraud probability (0-1)")
    label: str = Field(description="Human-readable label")
    timestamp: str


class BatchPredictionResponse(BaseModel):

    predictions: List[PredictionResponse]
    total: int
    fraud_count: int


class HealthResponse(BaseModel):

    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):

    model_name: str
    stage: str
    status: str
    timestamp: str


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time credit card fraud detection powered by an "
        "XGBoost model managed through MLflow. "
        "Part of the AIN-3009 MLOps Term Project."
    ),
    version="1.0.0",
)


@app.get("/", tags=["General"])
def root():

    return {
        "message": "Fraud Detection API — MLOps Term Project",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["General"])
def model_info():

    return ModelInfoResponse(
        model_name=MODEL_NAME,
        stage="Production",
        status="loaded" if model_loaded else "not loaded",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(transaction: Transaction):

    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    input_df = pd.DataFrame([transaction.features], columns=FEATURE_NAMES)
    prediction = model.predict(input_df)

    pred_val = int(prediction[0])

    try:
        unwrapped = model._model_impl
        if hasattr(unwrapped, "predict_proba"):
            prob = float(unwrapped.predict_proba(input_df)[0][1])
        else:
            prob = float(pred_val)
    except Exception:
        prob = float(pred_val)

    return PredictionResponse(
        prediction=pred_val,
        probability=round(prob, 6),
        label="Fraud" if pred_val == 1 else "Legitimate",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post(
    "/predict_batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
)
def predict_batch(batch: BatchTransactions):

    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    features_list = [t.features for t in batch.transactions]
    input_df = pd.DataFrame(features_list, columns=FEATURE_NAMES)
    predictions = model.predict(input_df)

    try:
        unwrapped = model._model_impl
        if hasattr(unwrapped, "predict_proba"):
            probs = unwrapped.predict_proba(input_df)[:, 1]
        else:
            probs = predictions.astype(float)
    except Exception:
        probs = predictions.astype(float)

    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probs)):
        pred_val = int(pred)
        results.append(
            PredictionResponse(
                prediction=pred_val,
                probability=round(float(prob), 6),
                label="Fraud" if pred_val == 1 else "Legitimate",
                timestamp=datetime.utcnow().isoformat(),
            )
        )

    fraud_count = sum(1 for r in results if r.prediction == 1)
    return BatchPredictionResponse(
        predictions=results,
        total=len(results),
        fraud_count=fraud_count,
    )


if __name__ == "__main__":
    host = config["serving"]["host"]
    port = config["serving"]["port"]
    print(f"\n[INFO] Starting server at http://{host}:{port}")
    print(f"[INFO] API docs at http://localhost:{port}/docs\n")
    uvicorn.run(app, host=host, port=port)
