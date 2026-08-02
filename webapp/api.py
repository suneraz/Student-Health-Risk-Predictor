"""
Student Health Risk Predictor - FastAPI backend
=================================================
Production-style API service serving the trained XGBoost model.

Run with:
    uvicorn api:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive Swagger UI documentation.

Needs xgboost_model.pkl in the same folder as this script.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_model.pkl")

# Load the model bundle ONCE at startup, not per-request.
bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
feature_cols = bundle["feature_columns"]
target_encoder = bundle.get("target_encoder")  # XGBoost needs this, Random Forest wouldn't

app = FastAPI(
    title="Student Health Risk Predictor API",
    description="Predicts student health risk (fit / at-risk / unhealthy) from "
                 "lifestyle and physiological data using a trained XGBoost model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RISK_INFO = {
    "fit": "Metrics align with a healthy lifestyle pattern.",
    "at-risk": "Some indicators suggest lifestyle adjustments could help.",
    "unhealthy": "Several indicators suggest elevated health risk. Consider consulting a professional.",
}


class StudentInput(BaseModel):
    """One student's lifestyle and physiological metrics."""
    sleep_duration: float = Field(..., ge=0, le=24, description="Hours of sleep per day")
    heart_rate: float = Field(..., ge=30, le=220, description="Resting heart rate (bpm)")
    bmi: float = Field(..., ge=10, le=60, description="Body mass index")
    calorie_expenditure: float = Field(..., ge=0, description="Calories burned per day")
    step_count: float = Field(..., ge=0, description="Daily step count")
    exercise_duration: float = Field(..., ge=0, description="Minutes of exercise per day")
    water_intake: float = Field(..., ge=0, le=15, description="Litres of water per day")
    diet_type: Literal["balanced", "non-veg", "veg", "missing"] = "balanced"
    stress_level: Literal["low", "medium", "high", "missing"] = "medium"
    sleep_quality: Literal["good", "average", "poor", "missing"] = "average"
    physical_activity_level: Literal["active", "moderate", "sedentary", "missing"] = "moderate"
    smoking_alcohol: Literal["no", "occasional", "yes", "missing"] = "no"
    gender: Literal["male", "female", "other", "missing"] = "male"

    class Config:
        json_schema_extra = {
            "example": {
                "sleep_duration": 7.5, "heart_rate": 72, "bmi": 22.0,
                "calorie_expenditure": 2400, "step_count": 9000, "exercise_duration": 40,
                "water_intake": 2.5, "diet_type": "balanced", "stress_level": "low",
                "sleep_quality": "good", "physical_activity_level": "active",
                "smoking_alcohol": "no", "gender": "male",
            }
        }


class PredictionResponse(BaseModel):
    prediction: str
    description: str
    probabilities: dict[str, float]


@app.get("/")
def root():
    """Health check / basic API info."""
    return {
        "service": "Student Health Risk Predictor API",
        "status": "running",
        "model": bundle["model_name"],
        "docs": "/docs",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentInput):
    """
    Predict health risk category for one student.

    Accepts the 13 lifestyle/physiological features as JSON, returns the
    predicted category (fit / at-risk / unhealthy) plus class probabilities.
    The pipeline's own preprocessing (OneHotEncoder for categorical fields)
    handles encoding internally, so raw values go straight in.
    """
    try:
        row = student.model_dump()
        X = pd.DataFrame([row])[feature_cols]

        pred = pipeline.predict(X)
        if target_encoder is not None:
            pred_label = target_encoder.inverse_transform(pred)[0]
        else:
            pred_label = pred[0]

        proba = pipeline.predict_proba(X)[0]
        classes = bundle["target_classes"]
        probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

        return PredictionResponse(
            prediction=pred_label,
            description=RISK_INFO.get(pred_label, ""),
            probabilities=probabilities,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
