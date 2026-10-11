import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "raw_data.csv"

df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("AI ENGINE CONDITION CLASSIFICATION")
print("=" * 60)

# --------------------------------------------------
# 2. Prepare features and target
# --------------------------------------------------

target_column = "Engine Condition"

X = df.drop(columns=[target_column])
y = df[target_column]

print("\nDataset shape:", df.shape)
print("Number of features:", X.shape[1])
print("Target distribution:")
print(y.value_counts())

# --------------------------------------------------
# 3. Split the data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 4. Define machine learning models
# --------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ),

    "Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        random_state=42,
    ),
}

# --------------------------------------------------
# 5. Train and evaluate
# --------------------------------------------------

results = []

for model_name, model in models.items():
    print("\n" + "=" * 60)
    print(f"Training: {model_name}")
    print("=" * 60)

    model.fit(X_train, y_train)
    model_path = PROJECT_DIR / "models" / (
    model_name.lower().replace(" ", "_") + ".joblib"
    )

    joblib.dump(model, model_path)

    print(f"Saved model: {model_path}")
    

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
    })

# --------------------------------------------------
# 6. Compare models
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df
    .sort_values("F1-score", ascending=False)
    .to_string(index=False)
)

# Save the comparison results
REPORT_FILE = PROJECT_DIR / "reports" / "model_comparison.csv"
results_df.to_csv(REPORT_FILE, index=False)

print(f"\nResults saved to: {REPORT_FILE}")