"""Step 8: Train-only harmonised preprocessing for OPTN DGF 70/10/20 split.

Run: python 8.feature_preprocessing_70_10_20.py
Uses pooled training data to fit one shared preprocessor (centralised preprocessing
statistics; NOT a privacy-preserving distributed statistics protocol).
Does not overwrite original processed data or split CSVs.
"""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CENTRES = ["21080", "14942", "07161", "25110", "23901"]
NUMERIC = ["AGE", "BMI_CALC", "CREAT_TRR", "AGE_DON", "BMI_DON_CALC", "CREAT_DON", "HLAMIS", "DAYSWAIT_CHRON", "DISTANCE", "NPKID", "COLD_ISCH_KI"]
CATEGORICAL = ["GENDER", "DIAB", "DIAL_TRR", "PREV_KI_TX", "GENDER_DON", "NON_HRT_DON", "ABO", "ABO_DON", "PUMP_KI"]
FEATURES = NUMERIC + CATEGORICAL
MISSING = {"", ".", "NA", "N/A", "NULL", "NONE", "NAN", "UNKNOWN", "U"}
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
    absent = sorted(set(required) - set(df.columns))
    if absent:
        raise ValueError(f"{path.name}: missing columns {absent}")
    for key in ("PT_CODE", "TRR_ID_CODE", "CTR_CODE"):
        df[key] = df[key].str.strip()
        if df[key].eq("").any():
            raise ValueError(f"{path.name}: blank {key}")
    labels = df["DGF"].str.strip()
    if not labels.isin(["0", "1"]).all():
        raise ValueError(f"{path.name}: invalid DGF labels")
    df["DGF"] = labels.astype(np.int64)
    for col in NUMERIC:
        s = df[col].str.strip()
        df[col] = pd.to_numeric(s.mask(s.str.upper().isin(MISSING)), errors="coerce")
    for col in CATEGORICAL:
        s = df[col].str.strip().str.upper()
        s = s.mask(s.isin(MISSING), "MISSING")
        unknown = sorted(set(s.unique()) - set(CATEGORIES[col]))
        if unknown:
            raise ValueError(f"{path.name}: unknown {col} codes {unknown}")
        df[col] = s
    return df


def make_preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("categorical", OneHotEncoder(
            categories=[CATEGORIES[c] for c in CATEGORICAL],
            handle_unknown="ignore", sparse_output=False, dtype=np.float32,
        ), CATEGORICAL),
    ], remainder="drop", sparse_threshold=0.0)


def check_ids(partitions):
    for name, frame in partitions.items():
        if frame.TRR_ID_CODE.duplicated().any():
            raise ValueError(f"Duplicate transplant IDs in {name}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        for key in ("PT_CODE", "TRR_ID_CODE"):
            if set(partitions[a][key]) & set(partitions[b][key]):
                raise ValueError(f"{key} overlap between {a} and {b}")


def save_npz(out, prefix, part, df, pre):
    X = np.asarray(pre.transform(df[FEATURES]), dtype=np.float32)
    y = df["DGF"].to_numpy(dtype=np.int64)
    if X.shape != (len(df), 54) or not np.isfinite(X).all():
        raise ValueError(f"Invalid processed shape/values for {prefix}_{part}: {X.shape}")
    np.savez_compressed(out / f"{prefix}_{part}.npz", X=X, y=y,
                        trr_id=df["TRR_ID_CODE"].to_numpy(),
                        centre=df["CTR_CODE"].to_numpy())
    return {"records": len(y), "positives": int(y.sum()), "dimensions": X.shape[1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(r"D:\ICC conference"))
    args = parser.parse_args()
    source = args.base / "centre_splits_70_10_20"
    out = args.base / "processed_70_10_20"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output folder not empty: {out}. Archive it before rerunning.")

    partitions = {p: read_clean(source / f"all_centres_{p}.csv") for p in ("train", "val", "test")}
    if [len(partitions[p]) for p in ("train", "val", "test")] != [21004, 3001, 6002]:
        raise ValueError("Unexpected split sizes; compare with Step 7 manifest")
    check_ids(partitions)
    for p, df in partitions.items():
        if not df.CTR_CODE.isin(CENTRES).all():
            raise ValueError(f"Unknown centre in {p}")

    # ONE shared preprocessing transform: fit on pooled TRAIN ONLY.
    # Do not fit on validation/test; do not fit separately per client.
    pre = make_preprocessor()
    pre.fit(partitions["train"][FEATURES])
    names = pre.get_feature_names_out().tolist()
    if len(names) != 54:
        raise ValueError(f"Expected 54 features, got {len(names)}")

    # Validate client partition membership against combined partition before writing.
    client_frames = {}
    for i, centre in enumerate(CENTRES, 1):
        for part in ("train", "val", "test"):
            frame = read_clean(source / f"client_{i}_{part}.csv")
            expected = partitions[part].loc[partitions[part].CTR_CODE == centre]
            if not frame.CTR_CODE.eq(centre).all() or len(frame) != len(expected):
                raise ValueError(f"Centre/count mismatch for client {i} {part}")
            if set(frame.TRR_ID_CODE) != set(expected.TRR_ID_CODE):
                raise ValueError(f"Transplant ID mismatch for client {i} {part}")
            client_frames[(i, part)] = frame

    out.mkdir(parents=True, exist_ok=False) if not out.exists() else None
    joblib.dump(pre, out / "shared_train_only_preprocessor.joblib")
    summary = {}
    for part, frame in partitions.items():
        summary[f"centralised_{part}"] = save_npz(out, "centralised", part, frame, pre)
    for i, centre in enumerate(CENTRES, 1):
        for part in ("train", "val", "test"):
            summary[f"client_{i}_{part}"] = save_npz(out, f"client_{i}", part, client_frames[(i, part)], pre)
            print(f"Client {i} ({centre}) {part}: {summary[f'client_{i}_{part}']}", flush=True)
    manifest = {
        "split": "70/10/20", "fit_partition": "pooled training only",
        "preprocessing": "shared pooled-training median imputation + StandardScaler; fixed one-hot schema",
        "privacy_caveat": "Pooled training records were used to compute shared preprocessing statistics. This preprocessing stage is not itself privacy-preserving or fully federated.",
        "features": {"numeric": NUMERIC, "categorical": CATEGORICAL,
                     "categorical_values": CATEGORIES, "encoded_names": names},
        "pending_checks": ["Verify DIAB=998 against OPTN dictionary", "Verify availability/timing of COLD_ISCH_KI and PUMP_KI"],
        "summary": summary,
    }
    (out / "feature_schema.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Patient/transplant overlap: ZERO", flush=True)
    print("Shared preprocessing fitted on TRAIN only; validation/test transformed only.", flush=True)
    print("Saved to:", out, flush=True)


if __name__ == "__main__":
    main()
