
import pandas as pd
from pathlib import Path

folder = Path(r"D:\ICC conference")
path = folder / "dgf_cohort.csv"

df = pd.read_csv(
    path,
    usecols=["TRR_ID_CODE", "PT_CODE", "DONOR_ID", "CTR_CODE", "DGF"],
    dtype=str,
    keep_default_na=False
)

for col in ["TRR_ID_CODE", "PT_CODE", "DONOR_ID", "CTR_CODE"]:
    df[col] = df[col].str.strip()

print("\n===== DUPLICATE VALIDATION =====")
print("Total records:", len(df))

print("\n1. Transplant record identifiers")
print("Unique TRR_ID_CODE:", df["TRR_ID_CODE"].nunique())
print("Duplicate TRR_ID_CODE rows:",
      df.duplicated(subset=["TRR_ID_CODE"]).sum())

print("\n2. Recipient identifiers")
print("Unique PT_CODE:", df["PT_CODE"].nunique())
print("Repeated recipient rows:",
      df.duplicated(subset=["PT_CODE"]).sum())

print("\n3. Donor identifiers")
print("Unique DONOR_ID:", df["DONOR_ID"].nunique())
print("Repeated donor rows:",
      df.duplicated(subset=["DONOR_ID"]).sum())

print("\n4. Exact duplicate records")
print("Exact duplicates:", df.duplicated().sum())

print("\n5. Conflicting transplant identifiers")
conflicts = df.groupby("TRR_ID_CODE").agg(
    patients=("PT_CODE", "nunique"),
    donors=("DONOR_ID", "nunique"),
    centres=("CTR_CODE", "nunique"),
    outcomes=("DGF", "nunique")
)

conflicts = conflicts[
    (conflicts > 1).any(axis=1)
]

print("TRR IDs with conflicting details:", len(conflicts))

print("\n6. Top 5 centres")
top5 = ["21080", "14942", "07161", "25110", "23901"]
selected = df[df["CTR_CODE"].isin(top5)]

print("Top 5 records:", len(selected))
print("Top 5 unique TRR IDs:", selected["TRR_ID_CODE"].nunique())
print("Top 5 duplicate TRR IDs:",
      selected.duplicated(subset=["TRR_ID_CODE"]).sum())
