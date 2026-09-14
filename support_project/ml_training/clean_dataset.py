"""
clean_dataset.py

Cleans the Bitext Customer Support dataset
and creates cleaned_dataset.csv.
"""

import re
import pandas as pd

from intent_mapping import get_category, get_priority


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

print("\n============================================")
print("DATASET LOADING")
print("============================================")

df = pd.read_csv("dataset.csv")

print("Raw rows:", len(df))
print("Columns:")
print(df.columns.tolist())


# ============================================================
# STEP 2: SHOW RAW CATEGORY DISTRIBUTION
# ============================================================

print("\n============================================")
print("RAW CATEGORY DISTRIBUTION")
print("============================================")

print(df["category"].value_counts())


# ============================================================
# STEP 3: KEEP ONLY REQUIRED COLUMNS
# ============================================================

df = df[
    [
        "instruction",
        "category",
        "intent"
    ]
]


# ============================================================
# STEP 4: REMOVE MISSING VALUES
# ============================================================

df = df.dropna(
    subset=[
        "instruction",
        "category",
        "intent"
    ]
)


# ============================================================
# STEP 5: TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text)

    # lowercase
    text = text.lower()

    # keep letters, numbers and spaces
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


df["message"] = df["instruction"].apply(clean_text)


# Remove extremely short messages

df = df[
    df["message"].str.len() > 5
]


# ============================================================
# STEP 6: MAP CATEGORY
# ============================================================

df["category"] = df["category"].apply(
    get_category
)


print("\nUnknown categories after mapping:")

print(
    (df["category"] == "unknown").sum()
)


# ============================================================
# STEP 7: CREATE PRIORITY
# ============================================================

df["priority"] = df["intent"].apply(
    get_priority
)


# ============================================================
# STEP 8: CREATE ACTION
# ============================================================

def decide_action(priority):

    if priority == "high":
        return "escalate"

    elif priority == "medium":
        return "draft"

    else:
        return "resolve"


df["action"] = df["priority"].apply(
    decide_action
)


# ============================================================
# STEP 9: CHECK DUPLICATES
# ============================================================

print("\nDuplicate messages before removal:")

print(
    df["message"].duplicated().sum()
)


df = df.drop_duplicates(
    subset=["message"]
)


# ============================================================
# STEP 10: FINAL DATASET
# ============================================================

final_df = df[
    [
        "message",
        "category",
        "intent",
        "priority",
        "action"
    ]
]


# ============================================================
# STEP 11: SAVE
# ============================================================

final_df.to_csv(
    "cleaned_dataset.csv",
    index=False
)


# ============================================================
# STEP 12: REPORT
# ============================================================

print("\n============================================")
print("CLEANING COMPLETE")
print("============================================")

print("Final rows:", len(final_df))


print("\nCATEGORY DISTRIBUTION")

print(
    final_df["category"].value_counts()
)


print("\nINTENT DISTRIBUTION")

print(
    final_df["intent"].value_counts()
)


print("\nPRIORITY DISTRIBUTION")

print(
    final_df["priority"].value_counts()
)


print("\nACTION DISTRIBUTION")

print(
    final_df["action"].value_counts()
)


print("\nMISSING VALUES")

print(
    final_df.isnull().sum()
)


print("\nDUPLICATE ROWS")

print(
    final_df.duplicated().sum()
)


print("\nSAMPLE DATA")

print(
    final_df.head(10).to_string()
)


print("\nSaved:")
print("cleaned_dataset.csv")