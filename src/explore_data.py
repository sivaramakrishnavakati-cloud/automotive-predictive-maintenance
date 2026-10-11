import pandas as pd
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "raw_data.csv"

print("=" * 60)
print("AUTOMOTIVE PREDICTIVE MAINTENANCE")
print("DATASET EXPLORATION")
print("=" * 60)

# Check that the dataset exists
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Make sure raw_data.csv is inside the data folder."
    )

# Load the dataset
df = pd.read_csv(DATA_FILE)

# Display basic information
print("\n1. DATASET SIZE")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n2. COLUMN NAMES")
print(df.columns.tolist())

print("\n3. FIRST FIVE RECORDS")
print(df.head())

print("\n4. DATA TYPES")
print(df.dtypes)

print("\n5. MISSING VALUES")
print(df.isnull().sum())

print("\n6. DUPLICATE ROWS")
print(df.duplicated().sum())

print("\n7. ENGINE CONDITION DISTRIBUTION")

if "Engine Condition" in df.columns:
    print(df["Engine Condition"].value_counts())
elif "engine_condition" in df.columns:
    print(df["engine_condition"].value_counts())
else:
    print("Target column not found. Check the column names above.")

print("\n8. NUMERICAL SUMMARY")
print(df.describe())

print("\nDataset exploration completed!")