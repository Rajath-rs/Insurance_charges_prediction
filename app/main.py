"""
Insurance Charges Prediction - FastAPI Backend
Run with: uvicorn app.main:app --reload
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional

# ─────────────────────────────────────────────
# Load artifacts at startup (once)
# ─────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent / "model"

with open(BASE / "best_model.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open(BASE / "scaler.pkl", "rb") as f:
    SCALER = pickle.load(f)

with open(BASE / "feature_cols.json") as f:
    FEATURE_COLS = json.load(f)

with open(BASE / "metadata.json") as f:
    METADATA = json.load(f)

NUM_COLS = METADATA["num_cols"]          # ["age", "bmi", "children"]
NEEDS_SCALING = METADATA["needs_scaling"]

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="Insurance Charges Prediction API",
    description="Predict medical insurance charges based on patient details.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Input / Output schemas
# ─────────────────────────────────────────────
class PatientInput(BaseModel):
    age:      int   = Field(..., ge=18, le=64,  example=35,    description="Age (18–64)")
    bmi:      float = Field(..., ge=10, le=60,  example=28.5,  description="Body Mass Index")
    children: int   = Field(..., ge=0,  le=5,   example=2,     description="Number of dependents")
    sex:      str   = Field(...,                example="male", description="male / female")
    smoker:   str   = Field(...,                example="yes",  description="yes / no")
    region:   str   = Field(...,                example="southeast",
                            description="northeast / northwest / southeast / southwest")

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v):
        v = v.lower().strip()
        if v not in ("male", "female"):
            raise ValueError("sex must be 'male' or 'female'")
        return v

    @field_validator("smoker")
    @classmethod
    def validate_smoker(cls, v):
        v = v.lower().strip()
        if v not in ("yes", "no"):
            raise ValueError("smoker must be 'yes' or 'no'")
        return v

    @field_validator("region")
    @classmethod
    def validate_region(cls, v):
        v = v.lower().strip()
        valid = {"northeast", "northwest", "southeast", "southwest"}
        if v not in valid:
            raise ValueError(f"region must be one of {valid}")
        return v


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    predicted_charge:     float
    predicted_charge_str: str
    model_used:           str
    model_r2:             float
    bmi_category:         str
    risk_category:        str
    confidence_range:     dict


class BatchInput(BaseModel):
    patients: list[PatientInput]


# ─────────────────────────────────────────────
# Helper: preprocess one patient dict → feature array
# ─────────────────────────────────────────────
def preprocess(data: PatientInput) -> np.ndarray:
    row = {col: 0 for col in FEATURE_COLS}

    row["age"]      = data.age
    row["bmi"]      = data.bmi
    row["children"] = data.children

    if data.sex == "male":
        row["sex_male"] = 1
    if data.smoker == "yes":
        row["smoker_yes"] = 1
    if data.region == "northwest":
        row["region_northwest"] = 1
    elif data.region == "southeast":
        row["region_southeast"] = 1
    elif data.region == "southwest":
        row["region_southwest"] = 1
    # northeast → all region dummies = 0 (reference category)

    df_row = pd.DataFrame([row])[FEATURE_COLS]

    if NEEDS_SCALING:
        df_row[NUM_COLS] = SCALER.transform(df_row[NUM_COLS])

    return df_row


def bmi_label(bmi: float) -> str:
    if bmi < 18.5: return "Underweight"
    if bmi < 25:   return "Normal"
    if bmi < 30:   return "Overweight"
    return "Obese"


def risk_label(charge: float, smoker: str, bmi: float) -> str:
    score = 0
    if smoker == "yes": score += 2
    if bmi >= 30:       score += 1
    if charge > 20000:  score += 1
    if score >= 3: return "High"
    if score >= 1: return "Medium"
    return "Low"


def predict_charge(data: PatientInput) -> dict:
    X = preprocess(data)
    log_pred = MODEL.predict(X)[0]
    charge   = float(np.exp(log_pred))

    # ±15 % confidence band (approximate)
    low  = round(charge * 0.85, 2)
    high = round(charge * 1.15, 2)

    return {
        "predicted_charge":     round(charge, 2),
        "predicted_charge_str": f"${charge:,.2f}",
        "model_used":           METADATA["best_model_name"],
        "model_r2":             METADATA["test_r2"],
        "bmi_category":         bmi_label(data.bmi),
        "risk_category":        risk_label(charge, data.smoker, data.bmi),
        "confidence_range":     {"low": low, "high": high},
    }


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    return {
        "message": "Insurance Charges Prediction API",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health", tags=["General"])
def health():
    return {
        "status":     "ok",
        "model":      METADATA["best_model_name"],
        "test_r2":    METADATA["test_r2"],
        "mae_usd":    METADATA["mae_usd"],
    }


@app.get("/model/info", tags=["Model"])
def model_info():
    """Returns full model comparison and metadata."""
    return METADATA


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: PatientInput):
    """
    Predict insurance charges for a single patient.

    - **age**: 18–64
    - **bmi**: Body Mass Index
    - **children**: 0–5
    - **sex**: male / female
    - **smoker**: yes / no
    - **region**: northeast / northwest / southeast / southwest
    """
    try:
        return predict_charge(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(batch: BatchInput):
    """Predict charges for multiple patients at once."""
    if len(batch.patients) > 100:
        raise HTTPException(status_code=400, detail="Max 100 patients per batch request.")
    results = []
    for i, patient in enumerate(batch.patients):
        try:
            result = predict_charge(patient)
            result["index"] = i
            results.append(result)
        except Exception as e:
            results.append({"index": i, "error": str(e)})
    return {"count": len(results), "predictions": results}


@app.get("/predict/example", tags=["Prediction"])
def example_prediction():
    """Returns a sample prediction so you can verify the API is working."""
    sample = PatientInput(
        age=35, bmi=28.5, children=2,
        sex="male", smoker="no", region="southeast"
    )
    result = predict_charge(sample)
    result["note"] = "This is a sample prediction for a 35-year-old non-smoker male."
    return result


@app.get("/stats/dataset", tags=["Analytics"])
def dataset_stats():
    """Returns key statistics from the training dataset."""
    return {
        "total_records":       1337,
        "features":            ["age", "sex", "bmi", "children", "smoker", "region"],
        "target":              "charges (USD)",
        "avg_charge":          13270.42,
        "min_charge":          1121.87,
        "max_charge":          63770.43,
        "smoker_percentage":   20.5,
        "avg_age":             39.2,
        "avg_bmi":             30.7,
    }