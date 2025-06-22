# api/app.py
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from .schema import CreditBatch, PredictionOut

# 直接用 joblib load 我們打包進 container 的 pipeline
PIPELINE_PATH = os.getenv("PIPELINE_PATH", "artifacts/model.joblib")
try:
    model = joblib.load(PIPELINE_PATH)
except Exception as e:
    raise RuntimeError(f"❌ 無法載入模型 {PIPELINE_PATH}\n{e}")

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
    try:
        df = pd.DataFrame([r.dict() for r in batch.records])
        proba = model.predict_proba(df)[:, 1]   # pipeline.predict_proba 回傳 Nx2
        preds = (proba >= THRESHOLD).astype(int)
        return [
            PredictionOut(probability=float(p), label=int(lbl))
            for p, lbl in zip(proba, preds)
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))