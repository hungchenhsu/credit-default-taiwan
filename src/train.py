"""
Run Optuna search + final training + MLflow logging.
python -m src.train
"""
import numpy as np 
import mlflow, optuna, joblib, lightgbm as lgb
import pathlib
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
from .data_prep import load_raw, clean
from .model_utils import build_preprocessor
from mlflow.models.signature import infer_signature

TRACK_URI = pathlib.Path(__file__).resolve().parents[1] / "mlruns"
mlflow.set_tracking_uri(TRACK_URI.as_uri())
mlflow.set_experiment("credit-risk-lgbm")

RANDOM_STATE = 42
TARGET = "default payment next month"
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)

df = clean(load_raw())
X, y = df.drop(columns=[TARGET]), df[TARGET]

cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
early_cb = lgb.early_stopping(50, verbose=False)

def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 64, 256),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True), 
        "n_estimators": trial.suggest_int("n_estimators", 800, 1500),              
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 4, 8),           
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "objective": "binary",
        "metric": "auc",
        "random_state": RANDOM_STATE,
    }

    aucs = []
    for tr_idx, val_idx in cv.split(X, y):
        pre = build_preprocessor(X)
        X_tr, X_val = pre.fit_transform(X.iloc[tr_idx]), pre.transform(X.iloc[val_idx])
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_tr, y.iloc[tr_idx],
                eval_set=[(X_val, y.iloc[val_idx])],
                callbacks=[early_cb])
        aucs.append(roc_auc_score(y.iloc[val_idx], clf.predict_proba(X_val)[:, 1]))
    return np.mean(aucs)

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40)
    best = study.best_params | {"objective": "binary", "metric": "auc",
                                "random_state": RANDOM_STATE}

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    pre = build_preprocessor(X)
    X_tr_p = pre.fit_transform(X_tr);  X_te_p = pre.transform(X_te)

    with mlflow.start_run(run_name="lgb_bin_final"):
        clf = lgb.LGBMClassifier(**best)
        clf.fit(X_tr_p, y_tr, eval_set=[(X_te_p, y_te)], callbacks=[early_cb])
        pipe = Pipeline([("pre", pre), ("clf", clf)])

        prob = pipe.predict_proba(X_te)[:, 1]
        f1   = f1_score(y_te, (prob >= 0.40).astype(int))
        auc  = roc_auc_score(y_te, prob)

        signature = infer_signature(
            X_te.head(),                    # example input
            pipe.predict(X_te.head())       # example output
        )

        mlflow.sklearn.log_model(
            sk_model       = pipe,
            artifact_path  = "model",       # UI 會顯示 Logged models ▸ model
            signature      = signature,
            input_example  = X_te.head(1)   # 讓 UI 顯示 JSON 範例
        )
        mlflow.log_metrics({"auc": auc, "f1": f1})
        joblib.dump(pipe, ARTIFACT_DIR / "model.joblib")

if __name__ == "__main__":
    main()                      