
import pandas as pd
from pathlib import Path

# ==========================================================
# STEP 4: FEATURE SELECTION AND MISSING VALUE AUDIT
# ==========================================================

folder = Path(r"D:\ICC conference")
input_file = folder / "dgf_cohort.csv"

top5_file = folder / "top5_dgf.csv"
report_file = folder / "feature_missing_report.csv"

TOP5 = ["21080", "14942", "07161", "25110", "23901"]

# Candidate features available before or around transplantation
numeric_features = [
    "AGE",
    "AGE_DON",
    "BMI_CALC",
    "BMI_DON_CALC",
    "CREAT_DON",
    "CREAT_TRR",
    "KDPI",
    "COLD_ISCH_KI",
    "DAYSWAIT_CHRON",
    "HLAMIS",
    "CURRENT_PRA",
    "DISTANCE",
    "NPKID"
]

categorical_features = [
    "GENDER",
    "GENDER_DON",
    "ABO",
    "ABO_DON",
    "DIAB",
    "DIAL_TRR",
    "DON_TY",
    "PUMP_KI",
    "NON_HRT_DON",
    "PREV_KI_TX"
]

id_columns = [
    "TRR_ID_CODE",
    "PT_CODE",
    "DONOR_ID",
    "CTR_CODE"
]

# Read header first to avoid errors from absent columns
available = pd.read_csv(input_file, nrows=0).columns.tolist()

features = numeric_features + categorical_features
present = [c for c in features if c in available]
absent = [c for c in features if c not in available]

required = id_columns + ["DGF"]

missing_required = [c for c in required if c not in available]
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

df = pd.read_csv(
    input_file,
    usecols=required + present,
    dtype=str,
    keep_default_na=False,
    low_memory=False
)

# Select the five highest-volume centres
df["CTR_CODE"] = df["CTR_CODE"].str.strip()
df = df[df["CTR_CODE"].isin(TOP5)].copy()

# Normalise missing tokens
missing_tokens = {
    "", ".", "NA", "N/A", "NULL",
    "NONE", "NAN", "UNKNOWN", "U"
}

for col in present:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].mask(
        df[col].str.upper().isin(missing_tokens)
    )

# Convert numerical features; malformed numbers become missing
for col in numeric_features:
    if col in present:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Save selected centres and candidate features
df.to_csv(top5_file, index=False)

# Feature-level audit
rows = []

for col in present:
    missing = int(df[col].isna().sum())
    total = len(df)

    rows.append({
        "Feature": col,
        "Type": "Numeric" if col in numeric_features else "Categorical",
        "Missing_Count": missing,
        "Missing_Percent": round(missing / total * 100, 2),
        "Unique_Values": df[col].nunique(dropna=True)
    })

report = pd.DataFrame(rows).sort_values(
    "Missing_Percent",
    ascending=False
)

report.to_csv(report_file, index=False)

print("\n========== TOP 5 DATASET ==========")
print("Records:", len(df))
print("Centres:", df["CTR_CODE"].nunique())
print("DGF positive:", (df["DGF"] == "1").sum())
print("DGF negative:", (df["DGF"] == "0").sum())

print("\n========== FEATURE MISSINGNESS ==========")
print(report.to_string(index=False))

print("\n========== ABSENT FEATURES ==========")
print(absent if absent else "None")

print("\n========== NUMERIC RANGE CHECK ==========")
for col in numeric_features:
    if col in df.columns:
        print(
            f"{col}: "
            f"min={df[col].min()}, "
            f"median={df[col].median()}, "
            f"max={df[col].max()}"
        )

print("\n========== SAVED FILES ==========")
print(top5_file)
print(report_file)
