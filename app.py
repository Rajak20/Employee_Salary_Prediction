import streamlit as st
import numpy as np
import pickle
from pathlib import Path

# Keep these files in the same folder as app.py
MODEL_PATH = Path(__file__).parent / "salary_model.pkl"
SCALER_PATH = Path(__file__).parent / "scaler (6).pkl"

st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    with open(SCALER_PATH, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

st.title("💼 Employee Salary Predictor")
st.write(
    "Estimate an employee's salary using a Linear Regression model "
    "and the StandardScaler fitted during model training."
)

try:
    model, scaler = load_artifacts()
except FileNotFoundError as error:
    st.error(
        f"Could not find `{error.filename}`. Place `app.py`, "
        "`salary_model.pkl`, and `scaler (6).pkl` in the same folder."
    )
    st.stop()
except Exception as error:
    st.error(f"Unable to load the model files: {error}")
    st.stop()

st.subheader("Enter employee details")

with st.form("salary_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        years_experience = st.number_input(
            "Years of experience", min_value=0, max_value=60, value=5, step=1
        )
        education_level = st.number_input(
            "Education level (encoded value)",
            min_value=0, max_value=20, value=2, step=1,
            help="Enter the numeric code used when training your model."
        )
        job_role = st.number_input(
            "Job role (encoded value)",
            min_value=0, max_value=100, value=1, step=1,
            help="Enter the numeric code used when training your model."
        )

    with col2:
        city_tier = st.number_input(
            "City tier (encoded value)",
            min_value=0, max_value=10, value=2, step=1,
            help="Enter the numeric code used when training your model."
        )
        performance_score = st.number_input(
            "Performance score", min_value=0.0, max_value=10.0,
            value=3.5, step=0.1
        )
        num_skills = st.number_input(
            "Number of skills", min_value=0, max_value=100, value=7, step=1
        )
        remote_work = st.selectbox(
            "Remote work",
            options=[0, 1],
            format_func=lambda value: "No (0)" if value == 0 else "Yes (1)"
        )

    submitted = st.form_submit_button(
        "Predict Salary", type="primary", use_container_width=True
    )

if submitted:
    # Feature order must exactly match scaler.feature_names_in_
    feature_names = [
        "age",
        "years_experience",
        "education_level",
        "job_role",
        "city_tier",
        "performance_score",
        "num_skills",
        "remote_work",
    ]
    input_data = np.array([[
        age,
        years_experience,
        education_level,
        job_role,
        city_tier,
        performance_score,
        num_skills,
        remote_work,
    ]], dtype=float)

    try:
        # Use a DataFrame so StandardScaler receives the original feature names.
        import pandas as pd
        input_df = pd.DataFrame(input_data, columns=feature_names)
        scaled_input = scaler.transform(input_df)
        prediction = float(model.predict(scaled_input)[0])

        st.success("Salary prediction completed!")
        st.metric("Predicted employee salary", f"₹{prediction:,.2f}")
        st.caption(
            "The displayed currency symbol assumes your training target is in INR. "
            "The prediction unit and scale are determined by the salary target "
            "used when training the model."
        )
    except Exception as error:
        st.error(f"Prediction failed: {error}")

with st.expander("Important: encoded categorical inputs"):
    st.write(
        "The scaler confirms that `education_level`, `job_role`, and `city_tier` "
        "are numeric features. The pickle files do not contain the original "
        "category-to-number mappings. Enter the same encoded values used in "
        "your training dataset; otherwise the prediction may be inaccurate."
    )
    st.write(
        "`remote_work` is passed as 0 or 1. The model expects the eight features "
        "in this exact order: age, years_experience, education_level, job_role, "
        "city_tier, performance_score, num_skills, remote_work."
    )

st.divider()
st.caption("Model: Linear Regression | Scaler: StandardScaler")
