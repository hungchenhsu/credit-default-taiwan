import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """衍生統計、趨勢與序列特徵。"""
    def fit(self, X, y=None): return self

    def transform(self, X):
        df = X.copy()
        pay_cols  = [c for c in df if c.startswith("PAY_")]
        bill_cols = [c for c in df if c.startswith("BILL_AMT")]
        pay_amt   = [c for c in df if c.startswith("PAY_AMT")]

        df["PAY_MAX"]   = df[pay_cols].max(1)
        df["PAY_MEAN"]  = df[pay_cols].mean(1)
        df["PAY_TREND"] = df.get("PAY_0", 0) - df.get("PAY_2", 0)

        for b, p in zip(bill_cols, pay_amt):
            df[f"NET_{b}"] = df[b] - df[p]

        df["UTIL_MEAN"]     = df[bill_cols].mean(1) / df["LIMIT_BAL"].replace(0, pd.NA)
        df["LIMIT_PER_AGE"] = df["LIMIT_BAL"] / df["AGE"].replace(0, pd.NA)

        if {"BILL_AMT1", "BILL_AMT6"}.issubset(df.columns):
            df["BILL_TREND"] = df["BILL_AMT1"] - df["BILL_AMT6"]

        if {"PAY_0", "PAY_2", "PAY_3"}.issubset(df.columns):
            df["PAY_LAST3_MEAN"] = df[["PAY_0", "PAY_2", "PAY_3"]].mean(1)
            df["PAY_WORSEN"]     = df["PAY_0"] - df[["PAY_2", "PAY_3"]].min(1)

        seq = df[pay_cols].astype(str).agg("-".join, 1)
        df["PAY_SEQ_CODE"] = pd.factorize(seq)[0]
        return df