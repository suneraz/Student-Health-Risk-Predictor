"""
Student Health Risk Predictor - Streamlit app
===============================================
Interactive web interface. Lets you pick between the three trained
comparison models: Neural Network, Random Forest, and XGBoost.

Run with:
    streamlit run streamlit_app.py

Needs neural_network_model.pkl, random_forest_model.pkl and xgboost_model.pkl
in the same folder as this script.
"""
import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Student Health Risk Predictor", page_icon="🩺", layout="wide")

MODEL_FILES = {
    "XGBoost": "xgboost_model.pkl",
    "Neural Network": "neural_network_model.pkl",
    "Random Forest": "random_forest_model.pkl",
}


@st.cache_resource
def load_models():
    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(BASE_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


models = load_models()

if not models:
    st.error("No model files found. Make sure neural_network_model.pkl, "
              "random_forest_model.pkl and xgboost_model.pkl are in the same folder as this script.")
    st.stop()

RISK_INFO = {
    "fit": {"color": "#2e7d32", "desc": "Metrics align with a healthy lifestyle pattern."},
    "at-risk": {"color": "#f57c00", "desc": "Some indicators suggest lifestyle adjustments could help."},
    "unhealthy": {"color": "#c62828", "desc": "Several indicators suggest elevated health risk. Consider consulting a professional."},
}

st.title("🩺 Student Health Risk Predictor")
st.caption("CIS 6005 · Computational Intelligence · Kaggle Playground Series S6E7")

model_choice = st.selectbox("Model", list(models.keys()))
bundle = models[model_choice]
pipeline = bundle["pipeline"]
feature_cols = bundle["feature_columns"]
cat_encoders = None  # not needed - the pipeline handles encoding internally
st.caption(f"Test accuracy: {bundle['test_accuracy']:.3f} · Macro F1: {bundle['test_macro_f1']:.3f}")

# pull dropdown category options from whichever model is loaded first,
# since all three were trained on the same categories
ref_bundle = list(models.values())[0]
cat_options = {}
for col in ref_bundle["categorical_features"]:
    # OneHotEncoder is the "cat" step inside the pipeline's preprocessor
    ohe = ref_bundle["pipeline"].named_steps["preprocess"].named_transformers_["cat"]
    idx = ref_bundle["categorical_features"].index(col)
    cat_options[col] = list(ohe.categories_[idx])

col_form, col_result = st.columns([1.3, 1])

with col_form:
    st.subheader("Your Metrics")

    st.markdown("**Physiological**")
    c1, c2, c3, c4 = st.columns(4)
    sleep_duration = c1.number_input("Sleep (hrs)", 0.0, 24.0, 7.0, 0.5)
    heart_rate = c2.number_input("Heart rate (bpm)", 30.0, 220.0, 75.0, 1.0)
    bmi = c3.number_input("BMI", 10.0, 60.0, 22.5, 0.1)
    water_intake = c4.number_input("Water (L/day)", 0.0, 15.0, 2.2, 0.1)

    st.markdown("**Activity**")
    c5, c6, c7, c8 = st.columns(4)
    step_count = c5.number_input("Steps/day", 0.0, 50000.0, 8000.0, 100.0)
    exercise_duration = c6.number_input("Exercise (min)", 0.0, 300.0, 30.0, 5.0)
    calorie_expenditure = c7.number_input("Calories", 0.0, 6000.0, 2300.0, 50.0)
    physical_activity_level = c8.selectbox("Activity level", cat_options["physical_activity_level"])

    st.markdown("**Lifestyle**")
    c9, c10, c11, c12, c13 = st.columns(5)
    diet_type = c9.selectbox("Diet", cat_options["diet_type"])
    stress_level = c10.selectbox("Stress", cat_options["stress_level"])
    sleep_quality = c11.selectbox("Sleep quality", cat_options["sleep_quality"])
    smoking_alcohol = c12.selectbox("Smoking/alcohol", cat_options["smoking_alcohol"])
    gender = c13.selectbox("Gender", cat_options["gender"])

    predict_clicked = st.button("Run Prediction", type="primary", use_container_width=True)

with col_result:
    st.subheader("Prediction")

    if predict_clicked:
        row = {
            "sleep_duration": sleep_duration, "heart_rate": heart_rate, "bmi": bmi,
            "calorie_expenditure": calorie_expenditure, "step_count": step_count,
            "exercise_duration": exercise_duration, "water_intake": water_intake,
            "diet_type": diet_type, "stress_level": stress_level,
            "sleep_quality": sleep_quality, "physical_activity_level": physical_activity_level,
            "smoking_alcohol": smoking_alcohol, "gender": gender,
        }

        # the pipeline's own OneHotEncoder step handles categorical columns,
        # so raw values go straight in - no manual encoding needed here
        X = pd.DataFrame([row])[feature_cols]
        pred = pipeline.predict(X)
        if "target_encoder" in bundle:
            pred_label = bundle["target_encoder"].inverse_transform(pred)[0]
        else:
            pred_label = pred[0]
        proba = pipeline.predict_proba(X)[0]
        classes = bundle["target_classes"]

        info = RISK_INFO.get(pred_label, {"color": "#333", "desc": ""})
        st.markdown(
            f"<div style='padding:10px 16px;border-radius:8px;background:{info['color']}22;"
            f"border:1px solid {info['color']};color:{info['color']};font-weight:700;"
            f"font-size:18px;display:inline-block;'>{pred_label.upper()}</div>",
            unsafe_allow_html=True,
        )
        st.write(info["desc"])
        st.markdown("**Class probabilities**")
        for cls, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
            st.write(f"{cls}: {p*100:.1f}%")
            st.progress(float(p))

        st.caption("Predictive estimate from a model trained on synthetic competition data — "
                    "not a medical assessment. Consult a healthcare professional for genuine concerns.")
    else:
        st.info("Fill in your metrics and click **Run Prediction** to see a risk category with class probabilities.")

    st.divider()
    st.caption(f"Model: {bundle['model_name']} · macro-F1 {bundle['test_macro_f1']:.3f} · 3-class")
