from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
DATA_FILE = PROJECT_DIR / "data" / "raw_data.csv"

FEATURES = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "Gradient Boosting": "gradient_boosting.joblib",
}


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Engine Condition Analyzer",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 AI Engine Condition Analyzer")

st.write(
    "An experimental machine learning application "
    "for analyzing engine sensor readings."
)

st.warning(
    "Research prototype only. The dataset's class labels "
    "have not yet been verified as healthy/faulty labels. "
    "Do not use these predictions for vehicle maintenance "
    "or safety decisions."
)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

if not DATA_FILE.exists():
    st.error("Dataset not found. Check the data folder.")
    st.stop()

df = pd.read_csv(DATA_FILE)

if not all(feature in df.columns for feature in FEATURES):
    st.error("The dataset does not contain the expected columns.")
    st.stop()


# --------------------------------------------------
# Project overview
# --------------------------------------------------

st.header("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", f"{len(df):,}")

with col2:
    st.metric("Sensor Measurements", len(FEATURES))

with col3:
    st.metric("Available Models", len(MODEL_FILES))

with st.expander("View sample dataset records"):
    st.dataframe(df.head(10), use_container_width=True)


# --------------------------------------------------
# Model selection
# --------------------------------------------------

st.header("Engine Condition Prediction")

model_name = st.selectbox(
    "Choose a machine learning model",
    list(MODEL_FILES.keys()),
)

model_path = MODEL_DIR / MODEL_FILES[model_name]

if not model_path.exists():
    st.error(f"Model file not found: {model_path.name}")
    st.stop()

model = joblib.load(model_path)


# --------------------------------------------------
# Sensor inputs
# --------------------------------------------------

st.subheader("Enter Sensor Measurements")

st.caption(
    "Initial values are the median measurements in the dataset. "
    "Input ranges are based on the dataset, not vehicle specifications."
)

defaults = df[FEATURES].median()

input_values = {}

col1, col2 = st.columns(2)

for index, feature in enumerate(FEATURES):
    column = col1 if index % 2 == 0 else col2

    with column:
        input_values[feature] = st.number_input(
            feature,
            min_value=float(df[feature].min()),
            max_value=float(df[feature].max()),
            value=float(defaults[feature]),
            format="%.4f",
        )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Engine Condition", type="primary"):

    input_df = pd.DataFrame(
        [input_values],
        columns=FEATURES,
    )

    prediction = int(model.predict(input_df)[0])

    st.subheader("Prediction Result")

    st.metric(
        "Predicted Dataset Class",
        str(prediction),
    )

    st.info(
        "This is the dataset's predicted class, not a confirmed "
        "diagnosis of a healthy or faulty engine."
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]

        probability_df = pd.DataFrame({
            "Dataset Class": [
                str(class_value)
                for class_value in model.classes_
            ],
            "Model Probability": probabilities,
        })

        st.subheader("Model Probability Estimates")

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            probability_df.set_index("Dataset Class")
        )

    st.subheader("Submitted Sensor Readings")

    st.dataframe(
        input_df,
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# Sensor visualization
# --------------------------------------------------

st.header("Sensor Data Analysis")

selected_sensor = st.selectbox(
    "Select a sensor to explore",
    FEATURES,
)

st.line_chart(
    df[selected_sensor].head(200).reset_index(drop=True)
)

st.caption(
    "This chart shows the first 200 records in file order. "
    "It is not a time-series trend unless chronological ordering "
    "has been established."
)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Portfolio project: Python | Pandas | Scikit-learn | Streamlit"
)