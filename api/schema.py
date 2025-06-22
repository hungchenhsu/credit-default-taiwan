# api/schema.py
from pydantic import BaseModel, Field, conint, confloat
from typing import List

# ------- 單筆輸入特徵 --------
class CreditFeatures(BaseModel):
    LIMIT_BAL: conint(ge=0) = Field(..., description="核發額度 (NT$)")
    SEX: conint(ge=1, le=2) = Field(..., description="性別 1=male 2=female")
    EDUCATION: conint(ge=1, le=4) = Field(..., description="最高學歷 1~4")
    MARRIAGE: conint(ge=1, le=3) = Field(..., description="婚姻狀態 1~3")
    AGE: conint(ge=18, le=100)
    # PAY_0, PAY_2, … PAY_6
    PAY_0:  conint(ge=-2, le=9)
    PAY_2:  conint(ge=-2, le=9)
    PAY_3:  conint(ge=-2, le=9)
    PAY_4:  conint(ge=-2, le=9)
    PAY_5:  conint(ge=-2, le=9)
    PAY_6:  conint(ge=-2, le=9)
    # BILL_AMT1–6 & PAY_AMT1–6
    BILL_AMT1: conint(ge=0)
    BILL_AMT2: conint(ge=0)
    BILL_AMT3: conint(ge=0)
    BILL_AMT4: conint(ge=0)
    BILL_AMT5: conint(ge=0)
    BILL_AMT6: conint(ge=0)
    PAY_AMT1:  conint(ge=0)
    PAY_AMT2:  conint(ge=0)
    PAY_AMT3:  conint(ge=0)
    PAY_AMT4:  conint(ge=0)
    PAY_AMT5:  conint(ge=0)
    PAY_AMT6:  conint(ge=0)

class CreditBatch(BaseModel):
    records: List[CreditFeatures]

class PredictionOut(BaseModel):
    probability: float = Field(..., example=0.82)
    label: int       = Field(..., example=1)