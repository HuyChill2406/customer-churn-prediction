"""FastAPI phục vụ mô hình churn.  Chạy local: uvicorn api:app --reload"""
import warnings
from contextlib import asynccontextmanager
from typing import List, Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", message="X does not have valid feature names")

MODEL_PATH = "model/churn_pipeline.joblib"
FEATURES = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_PATH)   # nạp 1 lần khi khởi động
    yield
    state.clear()


app = FastAPI(title="Bank Customer Churn API", version="1.0", lifespan=lifespan)


class Customer(BaseModel):
    CreditScore: int = Field(..., ge=300, le=850, examples=[650])
    Geography: Literal["France", "Germany", "Spain"]
    Gender: Literal["Female", "Male"]
    Age: int = Field(..., ge=18, le=100, examples=[40])
    Tenure: int = Field(..., ge=0, le=10, examples=[5])
    Balance: float = Field(..., ge=0, examples=[0.0])
    NumOfProducts: int = Field(..., ge=1, le=4, examples=[2])
    HasCrCard: Literal[0, 1]
    IsActiveMember: Literal[0, 1]
    EstimatedSalary: float = Field(..., ge=0, examples=[100000.0])


class Prediction(BaseModel):
    churn_probability: float
    high_risk: bool
    threshold: float


def _predict(customers: List[Customer], threshold: float) -> List[Prediction]:
    df = pd.DataFrame([c.model_dump() for c in customers])[FEATURES]
    proba = state["model"].predict_proba(df)[:, 1]
    return [Prediction(churn_probability=round(float(p), 4),
                       high_risk=bool(p >= threshold), threshold=threshold) for p in proba]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer, threshold: float = 0.5):
    return _predict([customer], threshold)[0]


@app.post("/predict_batch", response_model=List[Prediction])
def predict_batch(customers: List[Customer], threshold: float = 0.5):
    return _predict(customers, threshold)
