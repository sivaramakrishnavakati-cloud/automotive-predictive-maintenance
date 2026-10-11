import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_DIR / "data" / "raw_data.csv"
MODEL_FILE = PROJECT_DIR / "models" / "random_forest.joblib"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE)

# Separate features and target
X = df.drop(columns=["Engine Condition"])

# Load trained model
model = joblib.load(MODEL_FILE)

# Extract feature importance
importance = model.feature_importances_

# Create a DataFrame
importance_df = pd.DataFrame({
    "Sensor": X.columns,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Display results
print("\nSENSOR FEATURE IMPORTANCE")
print("=" * 50)

print(importance_df.to_string(index=False))

# Save results
importance_df.to_csv(
    REPORT_DIR / "feature_importance.csv",
    index=False
)

# Plot results
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Sensor"],
    importance_df["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Sensor")
plt.title("Engine Sensor Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "feature_importance.png",
    dpi=150
)

plt.show()

print("\nFeature analysis completed.")
print("Results saved in the reports folder.")