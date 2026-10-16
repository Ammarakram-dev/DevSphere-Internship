import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from pathlib import Path

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction System",
    page_icon="🔬",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "prediction_model.joblib"

st.title("🔬 Breast Cancer Diagnostic Prediction System")
st.write(
    "An interactive machine learning project demonstrating "
    "data-driven classification and model predictions."
)

st.warning(
    "Educational demonstration only. This model is not clinically "
    "validated and must not be used to diagnose or make medical decisions."
)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Run src/prediction_system.py first."
        )
    return joblib.load(MODEL_PATH)

try:
    bundle = load_model()
    model = bundle["model"]
    feature_names = list(bundle["feature_names"])
    target_names = list(bundle["target_names"])
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

# Dataset provides sensible input ranges and median starting values.
dataset = load_breast_cancer()
dataset_frame = pd.DataFrame(
    dataset.data,
    columns=dataset.feature_names
)

# --------------------------------------------------
# Prediction interface
# --------------------------------------------------
st.subheader("Enter the measurement values")
st.caption(
    "The fields start at the dataset medians. Replace them with "
    "values for an experiment. These inputs are not a substitute "
    "for measurements interpreted by a qualified clinician."
)

with st.form("prediction_form"):
    entered_values = {}

    # Organize the 30 features into three readable sections.
    groups = {
        "Mean measurements": [
            name for name in feature_names if name.startswith("mean ")
        ],
        "Measurement standard errors": [
            name for name in feature_names if name.endswith("error")
        ],
        "Worst measurements": [
            name for name in feature_names if name.startswith("worst ")
        ],
    }

    for group_title, group_features in groups.items():
        st.markdown(f"### {group_title}")
        columns = st.columns(3)

        for index, feature in enumerate(group_features):
            values = dataset_frame[feature]
            minimum = float(values.min())
            maximum = float(values.max())
            median = float(values.median())

            # Provide a small range padding so valid edge values
            # remain enterable without imposing a medical threshold.
            span = max(maximum - minimum, 0.01)
            lower_bound = max(0.0, minimum - span * 0.05)
            upper_bound = maximum + span * 0.05

            with columns[index % 3]:
                entered_values[feature] = st.number_input(
                    feature.replace(" ", " ").title(),
                    min_value=float(lower_bound),
                    max_value=float(upper_bound),
                    value=float(min(max(median, lower_bound), upper_bound)),
                    format="%.5f",
                    key=f"feature_{feature}"
                )

    submitted = st.form_submit_button(
        "Run Prediction",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# Display prediction
# --------------------------------------------------
if submitted:
    input_frame = pd.DataFrame(
        [[entered_values[name] for name in feature_names]],
        columns=feature_names
    )

    try:
        prediction = int(model.predict(input_frame)[0])
        predicted_label = str(target_names[prediction])

        st.divider()
        st.subheader("Prediction Result")

        if predicted_label.lower() == "malignant":
            st.error(f"Model classification: {predicted_label.title()}")
        else:
            st.success(f"Model classification: {predicted_label.title()}")

        st.caption(
            "This is a model-generated classification, not a medical diagnosis."
        )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_frame)[0]
            probability_table = pd.DataFrame({
                "Class": [str(name).title() for name in target_names],
                "Model-estimated probability": [
                    float(probability) * 100 for probability in probabilities
                ]
            })

            st.markdown("### Model-estimated class probabilities")
            st.dataframe(
                probability_table.style.format(
                    {"Model-estimated probability": "{:.2f}%"}
                ),
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                probability_table.set_index("Class")[
                    "Model-estimated probability"
                ]
            )

        with st.expander("Review submitted measurements"):
            st.dataframe(input_frame.T.rename(columns={0: "Input value"}))

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()
st.caption(
    "Project: DevSphere AI & ML Internship — Week 6 | "
    "Dataset: Scikit-learn Breast Cancer Wisconsin dataset"
)