import pandas as pd
from collections import Counter
from pathlib import Path

path = "KIDPAN_DATA_converted.csv"

output = Path(r"D:\ICC conference\identifier_cleaned.csv")

required = [
    "TRR_ID_CODE",
    "PT_CODE",
    "DONOR_ID"
]

audit = Counter()
first_chunk = True

missing_tokens = {"", ".", "UNKNOWN", "NULL", "NONE", "NAN", "NA"}

for chunk in pd.read_csv(
    path,
    dtype=str,
    chunksize=50000,
    keep_default_na=False,
    low_memory=False
):
    audit["original_rows"] += len(chunk)

    valid = pd.Series(True, index=chunk.index)

    for col in required:
        values = chunk[col].fillna("").str.strip()

        missing = values.str.upper().isin(missing_tokens)

        audit[f"missing_{col}"] += int(missing.sum())

        valid &= ~missing

    clean = chunk.loc[valid].copy()

    audit["valid_identifier_rows"] += len(clean)
    audit["removed_rows"] += len(chunk) - len(clean)

    clean.to_csv(
        output,
        mode="w" if first_chunk else "a",
        index=False,
        header=first_chunk
    )

    first_chunk = False

    print(
        f"Processed: {audit['original_rows']:,} | "
        f"Valid: {audit['valid_identifier_rows']:,}",
        end="\r"
    )

print("\n\nIDENTIFIER CLEANING AUDIT")

for key, value in audit.items():
    print(f"{key}: {value:,}")

print("\nSaved:", output)