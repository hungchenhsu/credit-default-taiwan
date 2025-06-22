import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from .feature_eng import CreditFeatureEngineer

def build_preprocessor(X, target_col="default payment next month"):
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    amt_cols = [c for c in X.columns if "AMT" in c]
    num_cols = [c for c in X.columns
                if c not in cat_cols + amt_cols + [target_col]]

    ct = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("amt_bins", KBinsDiscretizer(
            n_bins=20, encode="onehot-dense", strategy="quantile"), amt_cols),
        ("num", "passthrough", num_cols)
    ])
    return Pipeline([
        ("fe", CreditFeatureEngineer()),
        ("ct", ct)
    ])

def make_lgb_params(random_state=42, spw=6):
    return {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 128,
        "learning_rate": 0.05,
        "n_estimators": 1200,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": spw,
        "random_state": random_state,
    }