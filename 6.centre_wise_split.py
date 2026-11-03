
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

# ============================================================
# STEP 6: CENTRE-WISE PATIENT-DISJOINT TRAIN/TEST SPLIT
# ============================================================

BASE = Path(r"D:\ICC conference")
INPUT_FILE = BASE / "top5_dgf.csv"
OUTPUT_DIR = BASE / "centre_splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CENTRES = ["21080", "14942", "07161", "25110", "23901"]

TEST_SIZE = 0.20
N_SPLITS = 5
RANDOM_SEED = 42

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    dtype=str,
    keep_default_na=False,
    low_memory=False
)

required = ["TRR_ID_CODE", "PT_CODE", "CTR_CODE", "DGF"]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Required columns missing: {missing}")

for col in required:
    df[col] = df[col].str.strip()

if not df["CTR_CODE"].isin(CENTRES).all():
    raise ValueError("Unexpected centre codes found.")

if not df["DGF"].isin(["0", "1"]).all():
    raise ValueError("DGF labels must be 0 or 1.")

if df[required].eq("").any().any():
    raise ValueError("Blank required identifiers or labels found.")

if df["TRR_ID_CODE"].duplicated().any():
    raise ValueError("Duplicate transplant identifiers found.")

df["DGF"] = df["DGF"].astype(int)

print("\n========== ORIGINAL DATASET ==========")
print("Records:", len(df))
print("Unique patients:", df["PT_CODE"].nunique())
print("Centres:", df["CTR_CODE"].nunique())
print("DGF positive:", int(df["DGF"].sum()))
print("DGF negative:", int((df["DGF"] == 0).sum()))

# ============================================================
# 2. PATIENT-GROUPED STRATIFIED SPLIT
# ============================================================

# All records of a patient must belong to the same partition,
# even if the patient appears in multiple centres.
#
# StratifiedGroupKFold attempts to maintain label balance,
# while ensuring patient groups do not overlap between folds.

splitter = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_SEED
)

X = df[["CTR_CODE"]]
y = df["DGF"]
groups = df["PT_CODE"]

candidates = []

for fold_number, (train_idx, test_idx) in enumerate(
    splitter.split(X, y, groups),
    start=1
):
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    # Evaluate how closely this fold matches a
    # 20% test allocation in each centre.
    size_errors = []
    rate_errors = []

    for centre in CENTRES:
        full_c = df[df["CTR_CODE"] == centre]
        test_c = test[test["CTR_CODE"] == centre]

        if len(test_c) == 0:
            size_errors.append(1.0)
            rate_errors.append(1.0)
            continue

        observed_test_fraction = len(test_c) / len(full_c)

        size_errors.append(
            abs(observed_test_fraction - TEST_SIZE)
        )

        full_dgf_rate = full_c["DGF"].mean()
        test_dgf_rate = test_c["DGF"].mean()

        rate_errors.append(
            abs(test_dgf_rate - full_dgf_rate)
        )

    # Lower score = closer to desired proportions.
    score = np.mean(size_errors) + np.mean(rate_errors)

    candidates.append({
        "fold": fold_number,
        "score": score,
        "train_idx": train_idx,
        "test_idx": test_idx
    })

best = min(candidates, key=lambda item: item["score"])

train_df = df.iloc[best["train_idx"]].copy()
test_df = df.iloc[best["test_idx"]].copy()

print("\n========== SELECTED FOLD ==========")
print("Fold:", best["fold"])
print("Balance score:", round(best["score"], 6))

# ============================================================
# 3. LEAKAGE CHECKS
# ============================================================

train_patients = set(train_df["PT_CODE"])
test_patients = set(test_df["PT_CODE"])

patient_overlap = train_patients.intersection(test_patients)

train_transplants = set(train_df["TRR_ID_CODE"])
test_transplants = set(test_df["TRR_ID_CODE"])

transplant_overlap = train_transplants.intersection(test_transplants)

assert len(patient_overlap) == 0, "Patient leakage detected!"
assert len(transplant_overlap) == 0, "Transplant leakage detected!"

assert len(train_df) + len(test_df) == len(df)

print("\n========== LEAKAGE CHECK ==========")
print("Patient overlap:", len(patient_overlap))
print("Transplant overlap:", len(transplant_overlap))
print("All records accounted for: YES")

# ============================================================
# 4. SAVE INDIVIDUAL CENTRE FILES
# ============================================================

summary_rows = []

for i, centre in enumerate(CENTRES, start=1):

    full_c = df[df["CTR_CODE"] == centre]

    train_c = train_df[
        train_df["CTR_CODE"] == centre
    ].copy()

    test_c = test_df[
        test_df["CTR_CODE"] == centre
    ].copy()

    if len(train_c) == 0 or len(test_c) == 0:
        raise ValueError(
            f"Empty train/test partition for centre {centre}"
        )

    train_path = OUTPUT_DIR / f"client_{i}_train.csv"
    test_path = OUTPUT_DIR / f"client_{i}_test.csv"

    train_c.to_csv(train_path, index=False)
    test_c.to_csv(test_path, index=False)

    summary_rows.append({
        "Client": f"client_{i}",
        "Centre": centre,
        "Total": len(full_c),
        "Train": len(train_c),
        "Test": len(test_c),
        "Train_Percent": round(
            len(train_c) / len(full_c) * 100, 2
        ),
        "Test_Percent": round(
            len(test_c) / len(full_c) * 100, 2
        ),
        "Overall_DGF_Rate": round(
            full_c["DGF"].mean() * 100, 2
        ),
        "Train_DGF_Rate": round(
            train_c["DGF"].mean() * 100, 2
        ),
        "Test_DGF_Rate": round(
            test_c["DGF"].mean() * 100, 2
        ),
        "Train_DGF_Positive": int(train_c["DGF"].sum()),
        "Test_DGF_Positive": int(test_c["DGF"].sum())
    })

# ============================================================
# 5. SAVE COMBINED TRAIN AND TEST FILES
# ============================================================

train_df.to_csv(
    OUTPUT_DIR / "all_centres_train.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "all_centres_test.csv",
    index=False
)

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUTPUT_DIR / "split_summary.csv",
    index=False
)

print("\n========== CENTRE-WISE SPLIT SUMMARY ==========")
print(summary.to_string(index=False))

print("\n========== FINAL TOTALS ==========")
print("Training records:", len(train_df))
print("Testing records:", len(test_df))

print(
    "Overall train percentage:",
    round(len(train_df) / len(df) * 100, 2)
)

print(
    "Overall test percentage:",
    round(len(test_df) / len(df) * 100, 2)
)

print("Training DGF-positive:", int(train_df["DGF"].sum()))
print("Testing DGF-positive:", int(test_df["DGF"].sum()))

print("\n========== SAVED FILES ==========")
print("Folder:", OUTPUT_DIR)
print("5 client training files")
print("5 client testing files")
print("Combined train/test files")
print("split_summary.csv")

print("\nSTEP 6 COMPLETED SUCCESSFULLY")
