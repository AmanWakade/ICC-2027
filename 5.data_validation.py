
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# STEP 5: DATA VALIDATION BEFORE CENTRE-WISE TRAIN/TEST SPLIT
# ============================================================

BASE = Path(r"D:\ICC conference")

COHORT_FILE = BASE / "dgf_cohort.csv"
TOP5_FILE = BASE / "top5_dgf.csv"
SUMMARY_FILE = BASE / "centre_summary.csv"

TOP5 = ["21080", "14942", "07161", "25110", "23901"]

NUMERIC_FEATURES = [
    "AGE",
    "AGE_DON",
    "BMI_CALC",
    "BMI_DON_CALC",
    "CREAT_DON",
    "CREAT_TRR",
    "COLD_ISCH_KI",
    "DAYSWAIT_CHRON",
    "HLAMIS",
    "DISTANCE",
    "NPKID",
    "CURRENT_PRA"
]

MISSING_TOKENS = {
    "", ".", "NA", "N/A", "NULL",
    "NONE", "NAN", "UNKNOWN", "U"
}

# These are review flags, NOT automatic exclusion rules.
REVIEW_RANGES = {
    "AGE": (18, 100),
    "AGE_DON": (0, 100),
    "BMI_CALC": (10, 70),
    "BMI_DON_CALC": (10, 70),
    "CREAT_DON": (0.1, 15),
    "CREAT_TRR": (0.1, 40),
    "COLD_ISCH_KI": (0, 72),
    "DAYSWAIT_CHRON": (0, 15000),
    "HLAMIS": (0, 6),
    "DISTANCE": (0, 5000),
    "NPKID": (0, 10),
    "CURRENT_PRA": (0, 100)
}


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        low_memory=False
    )


def clean_text(series):
    s = series.astype(str).str.strip()
    return s.mask(s.str.upper().isin(MISSING_TOKENS))


def centre_counts(df, source_name):
    required = ["CTR_CODE", "DGF"]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"{source_name}: Missing column {col}"
            )

    work = df.copy()
    work["CTR_CODE"] = work["CTR_CODE"].str.strip()
    work["DGF"] = work["DGF"].str.strip()

    work = work[work["CTR_CODE"].isin(TOP5)]

    invalid = ~work["DGF"].isin(["0", "1"])

    if invalid.any():
        print(
            f"WARNING: {source_name} has "
            f"{invalid.sum()} invalid DGF labels."
        )

    work = work[~invalid].copy()

    result = (
        work.groupby("CTR_CODE")
        .agg(
            Records=("DGF", "size"),
            DGF_Positive=("DGF", lambda x: (x == "1").sum()),
            DGF_Negative=("DGF", lambda x: (x == "0").sum())
        )
        .reindex(TOP5, fill_value=0)
        .reset_index()
    )

    result["DGF_Rate"] = (
        result["DGF_Positive"]
        / result["Records"].replace(0, np.nan)
        * 100
    ).round(2)

    return result


print("\n" + "=" * 65)
print("STEP 5: DATA VALIDATION")
print("=" * 65)

cohort = load_csv(COHORT_FILE)
top5 = load_csv(TOP5_FILE)

# ============================================================
# 1. DGF COUNT RECONCILIATION
# ============================================================

print("\n========== 1. DGF COUNT RECONCILIATION ==========")

cohort_counts = centre_counts(cohort, "dgf_cohort.csv")
top5_counts = centre_counts(top5, "top5_dgf.csv")

comparison = cohort_counts.merge(
    top5_counts,
    on="CTR_CODE",
    suffixes=("_Cohort", "_Top5")
)

for metric in ["Records", "DGF_Positive", "DGF_Negative"]:
    comparison[f"{metric}_Difference"] = (
        comparison[f"{metric}_Cohort"]
        - comparison[f"{metric}_Top5"]
    )

print(comparison.to_string(index=False))

print("\nCOHORT TOP 5 TOTALS")
print(cohort_counts[
    ["Records", "DGF_Positive", "DGF_Negative"]
].sum().to_string())

print("\nTOP5 FILE TOTALS")
print(top5_counts[
    ["Records", "DGF_Positive", "DGF_Negative"]
].sum().to_string())

comparison.to_csv(
    BASE / "dgf_count_comparison.csv",
    index=False
)

# Compare with earlier centre_summary.csv
if SUMMARY_FILE.exists():
    summary = load_csv(SUMMARY_FILE)

    required_summary = {
        "CTR_CODE", "Transplants",
        "DGF_Positive", "DGF_Negative"
    }

    if required_summary.issubset(summary.columns):
        summary["CTR_CODE"] = (
            summary["CTR_CODE"].str.strip()
        )

        old = summary[
            summary["CTR_CODE"].isin(TOP5)
        ].copy()

        print("\nEARLIER CENTRE SUMMARY")
        print(
            old[
                ["CTR_CODE", "Transplants",
                 "DGF_Positive", "DGF_Negative"]
            ].to_string(index=False)
        )

        print("\nEARLIER SUMMARY TOTALS")
        for col in [
            "Transplants",
            "DGF_Positive",
            "DGF_Negative"
        ]:
            print(
                col,
                pd.to_numeric(
                    old[col], errors="coerce"
                ).sum()
            )
    else:
        print(
            "\ncentre_summary.csv columns differ; "
            "skipping old-summary comparison."
        )

# ============================================================
# 2. DUPLICATE AND PATIENT OVERLAP AUDIT
# ============================================================

print("\n========== 2. DUPLICATE AND PATIENT AUDIT ==========")

print("Total records:", len(top5))

for col in ["TRR_ID_CODE", "PT_CODE", "DONOR_ID"]:
    if col not in top5.columns:
        print(f"{col}: NOT AVAILABLE")
        continue

    values = clean_text(top5[col])
    valid = values.dropna()

    print(f"\n{col}")
    print("Missing IDs:", values.isna().sum())
    print("Unique nonmissing IDs:", valid.nunique())
    print("Repeated rows:", valid.duplicated().sum())

if "TRR_ID_CODE" in top5.columns:
    print(
        "\nExact duplicate rows:",
        top5.duplicated().sum()
    )

# Patients with multiple records may require group-aware splits
if "PT_CODE" in top5.columns:
    patients = clean_text(top5["PT_CODE"])

    counts = patients.dropna().value_counts()

    print("\nPatients with multiple transplant records:",
          (counts > 1).sum())

    print("Maximum records for one patient:",
          counts.max() if len(counts) else 0)

    # Number of patients appearing at multiple centres
    patient_centres = top5[
        ["PT_CODE", "CTR_CODE"]
    ].copy()

    patient_centres["PT_CODE"] = clean_text(
        patient_centres["PT_CODE"]
    )

    patient_centres = patient_centres.dropna(
        subset=["PT_CODE"]
    )

    multi_centre = (
        patient_centres.groupby("PT_CODE")["CTR_CODE"]
        .nunique()
    )

    print(
        "Patients appearing at multiple centres:",
        (multi_centre > 1).sum()
    )

# ============================================================
# 3. CENTRE-WISE MISSINGNESS
# ============================================================

print("\n========== 3. CENTRE-WISE MISSINGNESS ==========")

missing_rows = []

excluded = {
    "TRR_ID_CODE", "PT_CODE", "DONOR_ID",
    "CTR_CODE", "DGF", "FIRST_WK_DIAL"
}

features = [
    col for col in top5.columns
    if col not in excluded
]

top5["CTR_CODE"] = top5["CTR_CODE"].str.strip()

for centre in TOP5:
    part = top5[top5["CTR_CODE"] == centre]

    for feature in features:
        values = clean_text(part[feature])

        # Treat nonnumeric entries as missing in numeric fields
        if feature in NUMERIC_FEATURES:
            values = pd.to_numeric(
                values, errors="coerce"
            )

        missing = values.isna().sum()

        missing_rows.append({
            "Centre": centre,
            "Feature": feature,
            "Records": len(part),
            "Missing_Count": int(missing),
            "Missing_Percent": round(
                missing / len(part) * 100, 2
            ) if len(part) else np.nan
        })

missing_report = pd.DataFrame(missing_rows)

missing_report.to_csv(
    BASE / "centre_missingness.csv",
    index=False
)

print(
    missing_report.pivot(
        index="Feature",
        columns="Centre",
        values="Missing_Percent"
    ).round(2).to_string()
)

# ============================================================
# 4. NUMERICAL RANGE AND OUTLIER AUDIT
# ============================================================

print("\n========== 4. NUMERICAL RANGE CHECK ==========")

range_rows = []

for feature in NUMERIC_FEATURES:
    if feature not in top5.columns:
        continue

    values = pd.to_numeric(
        clean_text(top5[feature]),
        errors="coerce"
    )

    valid = values.dropna()

    if valid.empty:
        range_rows.append({
            "Feature": feature,
            "Valid": 0,
            "Missing": len(values),
            "Min": np.nan,
            "Median": np.nan,
            "Max": np.nan,
            "Outside_Review_Range": 0
        })
        continue

    low, high = REVIEW_RANGES[feature]

    outside = (
        (valid < low) | (valid > high)
    ).sum()

    range_rows.append({
        "Feature": feature,
        "Valid": len(valid),
        "Missing": int(values.isna().sum()),
        "Min": valid.min(),
        "Median": valid.median(),
        "Max": valid.max(),
        "Outside_Review_Range": int(outside)
    })

range_report = pd.DataFrame(range_rows)

range_report.to_csv(
    BASE / "numeric_range_report.csv",
    index=False
)

print(range_report.to_string(index=False))

# ============================================================
# 5. DATASET INTEGRITY
# ============================================================

print("\n========== 5. DATASET INTEGRITY ==========")

expected_centres = set(TOP5)
actual_centres = set(top5["CTR_CODE"].unique())

print("Expected centres:", len(expected_centres))
print("Actual centres:", len(actual_centres))
print("Centre codes match:",
      expected_centres == actual_centres)

if "DGF" in top5.columns:
    labels = top5["DGF"].str.strip()

    print("DGF label distribution:")
    print(labels.value_counts(dropna=False).to_string())

    print(
        "Only binary DGF labels:",
        labels.isin(["0", "1"]).all()
    )

# ============================================================
# FINISH
# ============================================================

print("\n========== REPORTS SAVED ==========")

for name in [
    "dgf_count_comparison.csv",
    "centre_missingness.csv",
    "numeric_range_report.csv"
]:
    print(BASE / name)

print("\nValidation completed.")
print("No source data was modified.")
