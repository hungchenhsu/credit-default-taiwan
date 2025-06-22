# api/app.py
import os
import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from .schema import CreditBatch, PredictionOut

# --------------------------- MLflow 設定 ---------------------------
# 1. Docker 內預設用 host.docker.internal:5001 連本機 MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:6060"))

MODEL_URI = "models:/credit_default_model@production"
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
except Exception as e:
    raise RuntimeError(f"❌ 無法載入 Production 模型 {MODEL_URI}\n{e}")

THRESHOLD = 0.40

app = FastAPI(
    title="Credit Default Prediction API",
    description="LightGBM + EasyEnsemble 模型 (thr=0.40)",
    version="1.0.0",
)

@app.get("/")
def root():
    return {"msg": "Credit-Default ML service is live 🔥"}

@app.post("/predict", response_model=list[PredictionOut])
def predict(batch: CreditBatch):
    """
    以陣列形式返回每筆 probability 及 label。
    """
    try:
        df = pd.DataFrame([r.dict() for r in batch.records])
        proba = model.predict(df)
        preds = (proba >= THRESHOLD).astype(int)
        return [
            PredictionOut(probability=float(p), label=int(lbl))
            for p, lbl in zip(proba, preds)
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))