import pandas as pd
from pathlib import Path

folder = Path(r"D:\ICC conference")

input_file = folder / "identifier_cleaned.csv"
cohort_file = folder / "dgf_cohort.csv"
summary_file = folder / "centre_summary.csv"

df = pd.read_csv(input_file, dtype=str, keep_default_na=False)

# Standardise categorical variables
for col in ["ORGAN", "MULTIORG", "DON_TY",
            "FIRST_WK_DIAL", "CTR_CODE"]:
    df[col] = df[col].str.strip().str.upper()

df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")

# Clinical eligibility
cohort = df[
    (df["ORGAN"] == "KI") &
    (df["MULTIORG"] != "Y") &
    (df["DON_TY"] == "C") &
    (df["AGE"] >= 18) &
    (df["FIRST_WK_DIAL"].isin(["Y", "N"])) &
    (~df["CTR_CODE"].isin(["", ".", "UNKNOWN", "NA", "NULL"]))
].copy()

cohort["DGF"] = (cohort["FIRST_WK_DIAL"] == "Y").astype(int)

cohort.to_csv(cohort_file, index=False)

# Centre-level summary
summary = cohort.groupby("CTR_CODE").agg(
    Transplants=("DGF", "size"),
    DGF_Positive=("DGF", "sum")
).reset_index()

summary["DGF_Negative"] = (
    summary["Transplants"] - summary["DGF_Positive"]
)

summary["DGF_Rate"] = (
    summary["DGF_Positive"] / summary["Transplants"] * 100
).round(2)

summary = summary.sort_values(
    "Transplants", ascending=False
)

summary.to_csv(summary_file, index=False)

print("\nFINAL DGF COHORT")
print("Total records:", len(cohort))
print("DGF positive:", cohort["DGF"].sum())
print("DGF negative:", (cohort["DGF"] == 0).sum())
print("Eligible centres:", len(summary))

print("\nTOP 10 CENTRES")
print(summary.head(10).to_string(index=False))

print("\nSaved:", cohort_file)
print("Saved:", summary_file)