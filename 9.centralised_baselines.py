"""Step 8: Train centralised Random Forest and XGBoost on frozen DGF splits.
Run after feature_preprocessing.py.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             recall_score, precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix)
from xgboost import XGBClassifier

BASE = Path(r"D:\ICC conference")
DATA = BASE / "processed"
OUT = BASE / "centralised_results"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

def load(name):
    with np.load(DATA / f"centralised_{name}.npz", allow_pickle=True) as d:
        return d["X"], d["y"], d["centre"]

def metrics(y, prob):
    pred = (prob >= 0.3).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": len(y), "positive": int(y.sum()),
        "roc_auc": roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": average_precision_score(y, prob) if len(np.unique(y)) == 2 else np.nan,
        "sensitivity": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "brier": brier_score_loss(y, prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }

def main():
    Xtr, ytr, ctr = load("train")
    Xte, yte, cte = load("test")
    assert Xtr.shape[0] == 24005 and Xte.shape[0] == 6002
    assert Xtr.shape[1] == Xte.shape[1]
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", n_jobs=-1,
            random_state=SEED
        ),
    }
    rows = []
    for name, model in models.items():
        print(f"\nTraining {name} ...", flush=True)
        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xte)[:, 1]
        row = {"model": name, "scope": "all_centres", **metrics(yte, prob)}
        rows.append(row)
        print(pd.DataFrame([row]).round(4).to_string(index=False))
        for centre in sorted(np.unique(cte)):
            mask = cte == centre
            rows.append({"model": name, "scope": str(centre), **metrics(yte[mask], prob[mask])})
        joblib.dump(model, OUT / f"{name}.joblib")
        pd.DataFrame({"trr_id": np.load(DATA / "centralised_test.npz", allow_pickle=True)["trr_id"],
                      "centre": cte, "label": yte, "probability": prob}).to_csv(
            OUT / f"{name}_test_predictions.csv", index=False)
    report = pd.DataFrame(rows)
    report.to_csv(OUT / "centralised_metrics.csv", index=False)
    print("\nCENTRE-WISE METRICS")
    print(report[["model", "scope", "n", "roc_auc", "pr_auc", "sensitivity", "f1", "brier"]].round(4).to_string(index=False))
    print("\nSaved:", OUT)
    print("Note: single seed, 0.5 threshold; no tuning or confidence intervals yet.")

if __name__ == "__main__":
    main()
