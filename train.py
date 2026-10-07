"""Train lại mô hình churn = LightGBM + StandardScaler + Dummies (đúng cấu hình final của nhóm).

Quy trình y hệt notebook:  bỏ id/CustomerId/Surname -> get_dummies(drop_first=True)
-> StandardScaler trên cả 11 cột -> LGBMClassifier(random_state=96, n_estimators=100).
Gộp tất cả vào MỘT Pipeline để deploy không bị lệch cột/scaler.

Chạy:  python train.py --train data/train.csv
"""
import argparse, json, os
import joblib, numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 96
NUM = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
       "HasCrCard", "IsActiveMember", "EstimatedSalary"]
CAT = ["Geography", "Gender"]
DROP = ["id", "CustomerId", "Surname"]
TARGET = "Exited"


def build_pipeline() -> Pipeline:
    # drop="first" + categories cố định  =>  Geography_Germany, Geography_Spain, Gender_Male
    # (giống pd.get_dummies(drop_first=True) trong notebook)
    pre = ColumnTransformer([
        ("num", "passthrough", NUM),
        ("cat", OneHotEncoder(categories=[["France", "Germany", "Spain"], ["Female", "Male"]],
                              drop="first", handle_unknown="ignore", sparse_output=False), CAT),
    ])
    model = LGBMClassifier(random_state=SEED, n_estimators=100, scale_pos_weight=1.0, verbose=-1)
    return Pipeline([("prep", pre), ("scale", StandardScaler()), ("model", model)])


def main(path: str, out: str):
    df = pd.read_csv(path).drop(columns=DROP, errors="ignore")
    X, y = df[NUM + CAT], df[TARGET].astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    proba = cross_val_predict(build_pipeline(), X, y, cv=skf, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "cv_auc": round(float(roc_auc_score(y, proba)), 4),
        "cv_f1_class1": round(f1_score(y, pred), 4),
        "cv_recall_class1": round(recall_score(y, pred), 4),
        "cv_precision_class1": round(precision_score(y, pred), 4),
        "n_rows": int(len(df)),
    }
    print("CV metrics:", metrics)

    pipe = build_pipeline().fit(X, y)          # train trên toàn bộ dữ liệu
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    joblib.dump(pipe, out)
    with open(os.path.join(os.path.dirname(out) or ".", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Đã lưu mô hình: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--out", default="model/churn_pipeline.joblib")
    a = ap.parse_args()
    main(a.train, a.out)
