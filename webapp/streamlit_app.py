"""
Student Health Risk Predictor - Streamlit app
===============================================
Interactive web interface. Lets you pick between the three trained
comparison models: Neural Network, Random Forest, and XGBoost.

Run with:
    streamlit run streamlit_app.py

Needs neural_network_model.pkl, random_forest_model.pkl and xgboost_model.pkl
in the same folder as this script, plus the .streamlit/config.toml theme file.
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

RISK_INFO = {
    "fit":       {"desc": "Metrics align with a healthy lifestyle pattern.", "icon": "✅"},
    "at-risk":   {"desc": "Some indicators suggest lifestyle adjustments could help.", "icon": "⚠️"},
    "unhealthy": {"desc": "Several indicators suggest elevated health risk. Consider consulting a professional.", "icon": "🚨"},
}

# Pearl-white background (set in .streamlit/config.toml) plus a set of
# design touches here: the page is stretched to use nearly the full
# viewport width, spacing/fonts are scaled up so nothing looks cramped,
# and cards get soft shadows, rounded corners, and light borders.
st.markdown("""
<style>
/* Use almost the full width of the screen, no more tiny centered column */
.block-container{
  padding-top: 1.4rem;
  padding-bottom: 1.2rem;
  padding-left: 3rem;
  padding-right: 3rem;
  max-width: 1800px;
}

/* Header */
.app-header{
  display: flex;
  align-items: center;
  gap: 18px;
  background: linear-gradient(135deg, #FFFFFF 0%, #F3F6FD 100%);
  border: 1px solid #ECE9E2;
  border-radius: 14px;
  padding: 18px 28px;
  box-shadow: 0 4px 14px rgba(0,0,0,0.05);
  margin-bottom: 1.1rem;
}
.app-header .icon-badge{
  font-size: 2.3rem;
  background: #EAF0FE;
  border-radius: 12px;
  width: 58px;
  height: 58px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.app-header h1{
  margin: 0;
  font-size: 1.9rem;
  line-height: 1.2;
}
.app-header .subtitle{
  color: #6B7280;
  font-size: 0.95rem;
  margin-top: 2px;
}

/* Section headers with icons */
.section-title{
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 1.25rem;
  font-weight: 700;
  margin-bottom: 0.6rem;
}
.section-title .badge{
  background: #EAF0FE;
  color: #2563EB;
  border-radius: 8px;
  width: 34px;
  height: 34px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.05rem;
}

/* Metric cards row */
div[data-testid="stMetric"]{
  background: #FAFAF8;
  border: 1px solid #ECE9E2;
  border-radius: 10px;
  padding: 10px 16px 6px 16px;
}
div[data-testid="stMetricValue"]{
  font-size: 1.5rem;
}

/* Card containers (form + result) */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background: #FFFFFF;
  border: 1px solid #ECE9E2;
  box-shadow: 0 3px 12px rgba(0,0,0,0.06);
  border-radius: 14px;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"]{
  padding: 6px 6px;
}

/* Slightly bigger captions acting as sub-section labels */
.stCaption, [data-testid="stCaptionContainer"]{
  font-size: 0.95rem !important;
  font-weight: 600;
  color: #374151 !important;
  margin-top: 4px;
}

/* Buttons */
.stButton>button, .stFormSubmitButton>button{
  border-radius: 8px;
  font-size: 1.02rem;
  padding: 0.6rem 1rem;
}

/* Prediction banner */
.pred-banner{
  border-radius: 12px;
  padding: 16px 20px;
  font-size: 1.05rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 14px;
}
.pred-fit{ background: #EAF7EE; color: #166534; border: 1px solid #BBE7C6; }
.pred-at-risk{ background: #FEF7E6; color: #92620A; border: 1px solid #F5DFA0; }
.pred-unhealthy{ background: #FDECEC; color: #991B1B; border: 1px solid #F5B8B8; }

.prob-row{
  display: flex;
  justify-content: space-between;
  font-size: 0.95rem;
  margin-top: 10px;
  margin-bottom: 2px;
}

.footer-note{
  text-align: center;
  color: #9CA3AF;
  font-size: 0.85rem;
  margin-top: 0.6rem;
}

hr{ margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


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

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown("""
<div class="app-header">
  <div class="icon-badge">🩺</div>
  <div>
    <h1>Student Health Risk Predictor</h1>
    <div class="subtitle">CIS 6005 · Computational Intelligence · Kaggle Playground Series S6E7</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Model selection
# ----------------------------------------------------------------------------
sel_col, m1, m2, m3 = st.columns([1.3, 1, 1, 1], gap="large")
model_choice = sel_col.selectbox("Select model", list(models.keys()))
bundle = models[model_choice]
pipeline = bundle["pipeline"]
feature_cols = bundle["feature_columns"]

m1.metric("Model", bundle["model_name"])
m2.metric("Test accuracy", f"{bundle['test_accuracy']*100:.1f}%")
m3.metric("Macro F1", f"{bundle['test_macro_f1']:.3f}")

st.divider()

# pull dropdown category options from whichever model is loaded first,
# since all three were trained on the same categories, and drop the
# placeholder "missing" category so it doesn't show up as a choice
ref_bundle = list(models.values())[0]
cat_options = {}
for col in ref_bundle["categorical_features"]:
    # OneHotEncoder is the "cat" step inside the pipeline's preprocessor
    ohe = ref_bundle["pipeline"].named_steps["preprocess"].named_transformers_["cat"]
    idx = ref_bundle["categorical_features"].index(col)
    cat_options[col] = [v for v in ohe.categories_[idx] if str(v).strip().lower() != "missing"]

# ----------------------------------------------------------------------------
# Metrics (left) and Prediction (right)
# ----------------------------------------------------------------------------
col_form, col_result = st.columns([1.5, 1], gap="large")

with col_form:
    st.markdown('<div class="section-title"><span class="badge">📋</span>Your Metrics</div>', unsafe_allow_html=True)

    with st.container(border=True):
        with st.form("metrics_form"):
            st.caption("❤️ Body & Vitals")
            c1, c2, c3 = st.columns(3)
            bmi = c1.slider("BMI", 10.0, 60.0, 22.5, 0.1)
            heart_rate = c2.slider("Resting heart rate (bpm)", 30.0, 220.0, 75.0, 1.0)
            gender = c3.selectbox("Gender", cat_options["gender"])

            st.caption("🏃 Activity & Energy")
            c4, c5, c6, c7, c8 = st.columns(5)
            step_count = c4.number_input("Daily steps", 0.0, 50000.0, 8000.0, 100.0)
            exercise_duration = c5.slider("Exercise (min)", 0.0, 300.0, 30.0, 5.0)
            calorie_expenditure = c6.number_input("Calories (kcal)", 0.0, 6000.0, 2300.0, 50.0)
            water_intake = c7.slider("Water (L/day)", 0.0, 15.0, 2.2, 0.1)
            physical_activity_level = c8.selectbox("Activity level", cat_options["physical_activity_level"])

            st.caption("😴 Sleep & Wellbeing")
            c9, c10, c11 = st.columns(3)
            sleep_duration = c9.slider("Sleep duration (hours)", 0.0, 24.0, 7.0, 0.5)
            sleep_quality = c10.selectbox("Sleep quality", cat_options["sleep_quality"])
            stress_level = c11.selectbox("Stress level", cat_options["stress_level"])

            st.caption("🍎 Diet & Habits")
            c12, c13 = st.columns(2)
            diet_type = c12.selectbox("Diet type", cat_options["diet_type"])
            smoking_alcohol = c13.selectbox("Smoking / alcohol use", cat_options["smoking_alcohol"])

            predict_clicked = st.form_submit_button("🔍 Run Prediction", type="primary", use_container_width=True)

with col_result:
    st.markdown('<div class="section-title"><span class="badge">🔮</span>Prediction</div>', unsafe_allow_html=True)

    with st.container(border=True):
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

            info = RISK_INFO.get(pred_label, {"desc": "", "icon": "ℹ️"})
            banner_class = f"pred-{pred_label}" if pred_label in RISK_INFO else "pred-at-risk"
            st.markdown(
                f'<div class="pred-banner {banner_class}">'
                f'<span style="font-size:1.4rem">{info["icon"]}</span>'
                f'<span>{pred_label.upper()} - {info["desc"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("**Class probabilities**")
            for cls, p in sorted(zip(classes, proba), key=lambda x: -x[1]):
                st.markdown(f'<div class="prob-row"><span>{cls}</span><span>{p*100:.1f}%</span></div>', unsafe_allow_html=True)
                st.progress(float(p))
        else:
            st.info("Fill in your metrics and click **Run Prediction** to see a risk category "
                     "with class probabilities.")

st.markdown(
    f'<div class="footer-note">Model: {bundle["model_name"]} &nbsp;·&nbsp; macro-F1 {bundle["test_macro_f1"]:.3f} &nbsp;·&nbsp; 3-class</div>',
    unsafe_allow_html=True,
)
