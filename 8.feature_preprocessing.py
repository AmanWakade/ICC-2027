"""Step 7: Leakage-safe preprocessing for 5 OPTN DGF clients and pooled baselines.
Run: python feature_preprocessing.py
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(r"D:\ICC conference")
SPLITS = BASE / "centre_splits"
OUT = BASE / "processed"
OUT.mkdir(parents=True, exist_ok=True)
CENTRES = ["21080", "14942", "07161", "25110", "23901"]
NUMERIC = ["AGE", "BMI_CALC", "CREAT_TRR", "AGE_DON", "BMI_DON_CALC", "CREAT_DON", "HLAMIS", "DAYSWAIT_CHRON", "DISTANCE", "NPKID", "COLD_ISCH_KI"]
CATEGORICAL = ["GENDER", "DIAB", "DIAL_TRR", "PREV_KI_TX", "GENDER_DON", "NON_HRT_DON", "ABO", "ABO_DON", "PUMP_KI"]
FEATURES = NUMERIC + CATEGORICAL
MISSING = {"", ".", "NA", "N/A", "NULL", "NONE", "NAN", "UNKNOWN", "U"}
# Fixed coding for all clients; unseen categories are mapped to MISSING.
# DIAB=998 is retained as a distinct code pending dictionary verification.
CATEGORIES = {
    "GENDER": ["F", "M", "MISSING"],
    "DIAB": ["1", "2", "3", "4", "5", "998", "MISSING"],
    "DIAL_TRR": ["N", "Y", "MISSING"],
    "PREV_KI_TX": ["N", "Y", "MISSING"],
    "GENDER_DON": ["F", "M", "MISSING"],
    "NON_HRT_DON": ["N", "Y", "MISSING"],
    "ABO": ["O", "A", "A1", "A2", "B", "AB", "A1B", "A2B", "MISSING"],
    "ABO_DON": ["O", "A", "A1", "A2", "B", "AB", "A1B", "A2B", "MISSING"],
    "PUMP_KI": ["N", "Y", "MISSING"],
}

def read_clean(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    required = FEATURES + ["DGF", "PT_CODE", "TRR_ID_CODE", "CTR_CODE"]
    missing_cols = sorted(set(required) - set(df.columns))
    if missing_cols:
        raise ValueError(f"{path.name}: missing columns {missing_cols}")
    df["CTR_CODE"] = df["CTR_CODE"].str.strip()
    df["PT_CODE"] = df["PT_CODE"].str.strip()
    df["TRR_ID_CODE"] = df["TRR_ID_CODE"].str.strip()
    if not df["DGF"].str.strip().isin(["0", "1"]).all():
        raise ValueError(f"Invalid DGF labels in {path}")
    df["DGF"] = df["DGF"].str.strip().astype(int)
    for col in NUMERIC:
        s = df[col].str.strip()
        s = s.mask(s.str.upper().isin(MISSING))
        df[col] = pd.to_numeric(s, errors="coerce")
    for col in CATEGORICAL:
        s = df[col].str.strip().str.upper()
        s = s.mask(s.isin(MISSING), "MISSING")
        # Explicit schema means no client-specific columns or categories.
        unknown = sorted(set(s.unique()) - set(CATEGORIES[col]))
        if unknown:
            raise ValueError(f"Unexpected {col} codes in {path.name}: {unknown}. Check OPTN dictionary before proceeding.")
        df[col] = s
    return df

def make_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(
        categories=[CATEGORIES[c] for c in CATEGORICAL],
        handle_unknown="ignore", sparse_output=False,
        dtype=np.float32,
    )
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ], remainder="drop", sparse_threshold=0.0)

def save_arrays(prefix, train, test, preprocessor):
    X_train = preprocessor.fit_transform(train[FEATURES]).astype(np.float32)
    X_test = preprocessor.transform(test[FEATURES]).astype(np.float32)
    y_train = train["DGF"].to_numpy(dtype=np.int64)
    y_test = test["DGF"].to_numpy(dtype=np.int64)
    if not (np.isfinite(X_train).all() and np.isfinite(X_test).all()):
        raise ValueError(f"Nonfinite processed features for {prefix}")
    np.savez_compressed(OUT / f"{prefix}_train.npz", X=X_train, y=y_train,
                        trr_id=train["TRR_ID_CODE"].to_numpy(), centre=train["CTR_CODE"].to_numpy())
    np.savez_compressed(OUT / f"{prefix}_test.npz", X=X_test, y=y_test,
                        trr_id=test["TRR_ID_CODE"].to_numpy(), centre=test["CTR_CODE"].to_numpy())
    joblib.dump(preprocessor, OUT / f"{prefix}_preprocessor.joblib")
    return X_train.shape[1], len(train), len(test)

def main():
    # Check for cross-split patient overlap before any fitting.
    all_train = read_clean(SPLITS / "all_centres_train.csv")
    all_test = read_clean(SPLITS / "all_centres_test.csv")
    assert set(all_train.PT_CODE).isdisjoint(set(all_test.PT_CODE)), "Patient leakage"
    assert set(all_train.TRR_ID_CODE).isdisjoint(set(all_test.TRR_ID_CODE)), "Transplant leakage"
    assert len(all_train) == 24005 and len(all_test) == 6002, "Unexpected split sizes"
    expected_names = None
    for i, centre in enumerate(CENTRES, 1):
        train = read_clean(SPLITS / f"client_{i}_train.csv")
        test = read_clean(SPLITS / f"client_{i}_test.csv")
        if not (train.CTR_CODE.eq(centre).all() and test.CTR_CODE.eq(centre).all()):
            raise ValueError(f"Centre mismatch in client {i}")
        pre = make_preprocessor()
        n_features, ntrain, ntest = save_arrays(f"client_{i}", train, test, pre)
        names = pre.get_feature_names_out().tolist()
        if expected_names is None:
            expected_names = names
        elif names != expected_names:
            raise ValueError("Inconsistent feature ordering across clients")
        print(f"Client {i} ({centre}): train={ntrain}, test={ntest}, encoded_features={n_features}")
    # Separate pooled training-only preprocessing for centralised baselines.
    pre = make_preprocessor()
    n_features, ntrain, ntest = save_arrays("centralised", all_train, all_test, pre)
    if pre.get_feature_names_out().tolist() != expected_names:
        raise ValueError("Centralised feature schema differs from clients")
    (OUT / "feature_schema.json").write_text(json.dumps({
        "numeric": NUMERIC, "categorical": CATEGORICAL,
        "categorical_values": CATEGORIES, "encoded_names": expected_names,
        "target": "DGF", "prediction_time": "transplant completion, before post-transplant dialysis",
        "note": "DIAB 998 meaning and perioperative timing require dictionary confirmation"
    }, indent=2), encoding="utf-8")
    print(f"Centralised: train={ntrain}, test={ntest}, encoded_features={n_features}")
    print("Patient overlap: 0; source files unchanged")
    print("Saved to:", OUT)

if __name__ == "__main__":
    main()
