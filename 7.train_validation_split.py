"""Step 7: Create a patient-disjoint ~70/10/20 train/validation/test split.

Input: existing centre_splits/all_centres_train.csv and all_centres_test.csv.
The existing test set is NEVER modified; original inputs are never overwritten.

Run: python 7.train_validation_split.py
Optional: python 7.train_validation_split.py --base "D:\\ICC conference"
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

CENTRES = ["21080", "14942", "07161", "25110", "23901"]
REQUIRED = ["PT_CODE", "TRR_ID_CODE", "CTR_CODE", "DGF"]


def read_split(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    absent = set(REQUIRED) - set(df.columns)
    if absent:
        raise ValueError(f"{path}: missing {sorted(absent)}")
    for c in REQUIRED:
        df[c] = df[c].str.strip()
    if df[REQUIRED].eq("").any().any():
        raise ValueError(f"{path}: blank patient/transplant/centre/label")
    if not df.DGF.isin(["0", "1"]).all():
        raise ValueError(f"{path}: invalid DGF labels")
    if not df.CTR_CODE.isin(CENTRES).all():
        raise ValueError(f"{path}: unexpected centre code")
    return df


def check_disjoint(a, b, name_a, name_b):
    for key in ("PT_CODE", "TRR_ID_CODE"):
        overlap = set(a[key]) & set(b[key])
        if overlap:
            raise ValueError(f"{name_a}/{name_b}: {len(overlap)} overlapping {key} values")


def choose_split(train, seed, attempts):
    """Choose among patient-grouped 8-fold candidates (1/8 = 12.5% of 80%)."""
    y = train.DGF.astype(int).to_numpy()
    groups = train.PT_CODE.to_numpy()
    target = 0.125
    candidates = []
    for offset in range(attempts):
        splitter = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed + offset)
        for fold, (tr_idx, val_idx) in enumerate(splitter.split(train[["CTR_CODE"]], y, groups), 1):
            tr, val = train.iloc[tr_idx], train.iloc[val_idx]
            if any((tr.CTR_CODE == c).sum() == 0 or (val.CTR_CODE == c).sum() == 0 for c in CENTRES):
                continue
            if any(val.loc[val.CTR_CODE == c, "DGF"].nunique() < 2 for c in CENTRES):
                continue
            size_err, rate_err = [], []
            for c in CENTRES:
                full_c = train[train.CTR_CODE == c]
                val_c = val[val.CTR_CODE == c]
                size_err.append(abs(len(val_c) / len(full_c) - target))
                rate_err.append(abs(val_c.DGF.astype(int).mean() - full_c.DGF.astype(int).mean()))
            global_size_err = abs(len(val) / len(train) - target)
            score = float(np.mean(size_err) + np.mean(rate_err) + global_size_err)
            candidates.append((score, offset, fold, tr_idx, val_idx))
    if not candidates:
        raise RuntimeError("No feasible grouped validation split found")
    score, offset, fold, tr_idx, val_idx = min(candidates, key=lambda item: item[:3])
    return train.iloc[tr_idx].copy(), train.iloc[val_idx].copy(), {"score": score, "seed": seed + offset, "fold": fold}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(r"D:\ICC conference"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attempts", type=int, default=10)
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("--attempts must be >= 1")
    source = args.base / "centre_splits"
    dest = args.base / "centre_splits_70_10_20"
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"Output folder already contains files: {dest}. Archive or choose another location.")
    original_train = read_split(source / "all_centres_train.csv")
    test = read_split(source / "all_centres_test.csv")
    if len(original_train) != 24005 or len(test) != 6002:
        raise ValueError(f"Unexpected original sizes: train={len(original_train)}, test={len(test)}")
    if original_train.TRR_ID_CODE.duplicated().any() or test.TRR_ID_CODE.duplicated().any():
        raise ValueError("Duplicate transplant IDs")
    check_disjoint(original_train, test, "original_train", "test")
    train, val, selected = choose_split(original_train, args.seed, args.attempts)
    check_disjoint(train, val, "train", "validation")
    check_disjoint(train, test, "train", "test")
    check_disjoint(val, test, "validation", "test")
    assert len(train) + len(val) + len(test) == len(original_train) + len(test)
    assert set(train.TRR_ID_CODE) | set(val.TRR_ID_CODE) == set(original_train.TRR_ID_CODE)
    for c in CENTRES:
        if any(d.loc[d.CTR_CODE == c, "DGF"].nunique() < 2 for d in (train, val, test)):
            raise ValueError(f"Centre {c} lacks both DGF classes in one partition")
    dest.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("val", val), ("test", test)):
        frame.to_csv(dest / f"all_centres_{name}.csv", index=False)
        for i, c in enumerate(CENTRES, 1):
            frame[frame.CTR_CODE == c].to_csv(dest / f"client_{i}_{name}.csv", index=False)
    summary = []
    for i, c in enumerate(CENTRES, 1):
        for part, frame in (("train", train), ("val", val), ("test", test)):
            d = frame[frame.CTR_CODE == c]
            summary.append({"client": i, "centre": c, "partition": part,
                            "records": len(d), "dgf_positive": int(d.DGF.astype(int).sum()),
                            "dgf_rate": float(d.DGF.astype(int).mean())})
    pd.DataFrame(summary).to_csv(dest / "split_summary.csv", index=False)
    manifest = {"requested_ratio": [0.70, 0.10, 0.20], "selection": selected,
                "counts": {"train": len(train), "validation": len(val), "test": len(test)},
                "note": "Original test held unchanged; 8-fold patient-grouped validation approximates 12.5% of original training records."}
    (dest / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Saved to:", dest)
    print(json.dumps(manifest, indent=2))
    print("Patient and transplant ID overlap: zero across train/validation/test")


if __name__ == "__main__":
    main()
