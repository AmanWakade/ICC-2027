
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"D:\ICC conference\centre_splits")

CENTRES = {
    1: "21080",
    2: "14942",
    3: "07161",
    4: "25110",
    5: "23901"
}

NUMERIC = [
    "AGE", "BMI_CALC", "CREAT_TRR",
    "AGE_DON", "BMI_DON_CALC", "CREAT_DON",
    "HLAMIS", "DAYSWAIT_CHRON",
    "DISTANCE", "NPKID", "COLD_ISCH_KI"
]

CATEGORICAL = [
    "GENDER", "DIAB", "DIAL_TRR", "PREV_KI_TX",
    "GENDER_DON", "NON_HRT_DON",
    "ABO", "ABO_DON", "PUMP_KI"
]

# These variables need confirmation of availability
# at the intended clinical prediction time.
TIMING_REVIEW = {
    "COLD_ISCH_KI": "Available after cold ischaemia duration is known",
    "PUMP_KI": "Check whether recorded before prediction time",
    "CREAT_TRR": "Confirm measurement is pre-transplant",
    "HLAMIS": "Confirm matching information is available",
    "DISTANCE": "Confirm donor-recipient logistics timing"
}

MISSING = {
    "", ".", "NA", "N/A", "NULL",
    "NONE", "NAN", "UNKNOWN", "U"
}

categorical_rows = []
numeric_rows = []
timing_rows = []

for client_id, centre in CENTRES.items():

    path = BASE / f"client_{client_id}_train.csv"

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False
    )

    print(f"\n========== CLIENT {client_id}: {centre} ==========")
    print("Training records:", len(df))

    for feature in CATEGORICAL:

        if feature not in df.columns:
            print("MISSING COLUMN:", feature)
            continue

        values = df[feature].astype(str).str.strip()
        values = values.mask(values.str.upper().isin(MISSING))

        counts = values.value_counts(dropna=False)

        print(f"\n{feature}")
        print(counts.head(15).to_string())

        categorical_rows.append({
            "Client": client_id,
            "Centre": centre,
            "Feature": feature,
            "Missing_Percent": round(
                values.isna().mean() * 100, 2
            ),
            "Categories": " | ".join(
                sorted(values.dropna().unique())
            )
        })

    for feature in NUMERIC:

        if feature not in df.columns:
            print("MISSING COLUMN:", feature)
            continue

        values = df[feature].astype(str).str.strip()
        values = values.mask(values.str.upper().isin(MISSING))
        values = pd.to_numeric(values, errors="coerce")

        numeric_rows.append({
            "Client": client_id,
            "Centre": centre,
            "Feature": feature,
            "Missing_Percent": round(
                values.isna().mean() * 100, 2
            ),
            "Min": values.min(),
            "Median": values.median(),
            "Max": values.max()
        })

    for feature, concern in TIMING_REVIEW.items():
        timing_rows.append({
            "Client": client_id,
            "Centre": centre,
            "Feature": feature,
            "Timing_Concern": concern,
            "Present": feature in df.columns
        })

pd.DataFrame(categorical_rows).to_csv(
    BASE / "categorical_audit.csv",
    index=False
)

pd.DataFrame(numeric_rows).to_csv(
    BASE / "numeric_feature_audit.csv",
    index=False
)

pd.DataFrame(timing_rows).to_csv(
    BASE / "prediction_timing_review.csv",
    index=False
)

print("\n========== NUMERIC FEATURE SUMMARY ==========")
numeric_report = pd.DataFrame(numeric_rows)
print(numeric_report.to_string(index=False))

print("\n========== PREDICTION TIMING REVIEW ==========")
print(pd.DataFrame(timing_rows).drop_duplicates(
    subset=["Feature"]
)[["Feature", "Timing_Concern"]].to_string(index=False))

print("\nReports saved in:", BASE)
print("No source data modified.")
