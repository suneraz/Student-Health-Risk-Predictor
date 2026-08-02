# Student Health Risk Predictor — Project Files

CIS 6005 Computational Intelligence — WRIT1 Mini Project
Kaggle competition: [Playground Series S6E7 — Predicting Student Health Risk](https://www.kaggle.com/competitions/playground-series-s6e7)

**Note:** `submission.csv` (for your Kaggle leaderboard upload) is provided as a
separate download, not inside this package.

## What's in here

```
Student_Health_Risk_Report.docx/.pdf   ← YOUR REPORT — submit this to Moodle/Turnitin
train_model.py                          ← full production pipeline (LightGBM, all 690k rows)

model_artifacts/
    final_model.pkl                     ← trained production LightGBM model
    preprocessing.pkl                   ← fitted encoders/medians both apps need

webapp/
    api.py                              ← FastAPI service (POST /predict, auto Swagger docs)
    streamlit_app.py                    ← Streamlit interactive web interface
    final_model.pkl / preprocessing.pkl ← local copies so each app runs standalone

notebooks/
    neural_network_model.ipynb          ← Neural Network (MLP), standalone
    random_forest_model.ipynb           ← Random Forest, standalone
    xgboost_model.ipynb                 ← XGBoost, standalone

comparison_models/
    neural_network_model.pkl            ← saved Neural Network (96.4% acc, F1 0.900)
    random_forest_model.pkl             ← saved Random Forest (94.6% acc, F1 0.875)
    xgboost_model.pkl                   ← saved XGBoost (96.6% acc, F1 0.906)
    load_and_predict_example.py         ← shows how to load and use any of the three
```

## Before you submit — two things only you can do

1. **Kaggle leaderboard proof.** Upload the separately-provided `submission.csv` to
   the competition yourself. Then open the report, find the red dashed box in
   Section 6.8, and replace it with your Kaggle username, public leaderboard score,
   and a screenshot.
2. **Screenshots of both apps.** Run both apps (steps below), screenshot the FastAPI
   Swagger docs page and a completed Streamlit prediction, and drop them into the two
   red dashed boxes in Section 6.10 of the report.

## Running the FastAPI service

```
cd webapp
pip3 install fastapi uvicorn pandas scikit-learn lightgbm joblib
uvicorn api:app --reload --port 8000
```
Open **http://localhost:8000/docs** for interactive Swagger documentation — you can
test the `/predict` endpoint directly from that page without writing any code.

Example request (from a second terminal, while the server is running):
```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "sleep_duration": 7.5, "heart_rate": 72, "bmi": 22.0, "calorie_expenditure": 2400,
  "step_count": 9000, "exercise_duration": 40, "water_intake": 2.5,
  "diet_type": "balanced", "stress_level": "low", "sleep_quality": "good",
  "physical_activity_level": "active", "smoking_alcohol": "no", "gender": "male"
}'
```

## Running the Streamlit app

```
cd webapp
pip3 install streamlit pandas scikit-learn lightgbm joblib
streamlit run streamlit_app.py
```
Opens automatically in your browser (usually **http://localhost:8501**).

Note: the two apps are independent — each loads the model directly from its own
copy of `final_model.pkl`/`preprocessing.pkl` in the `webapp/` folder. You don't
need both running at once; run whichever one you're demonstrating.

## Running the notebooks (for your viva)

Copy `train.csv` from your original Kaggle download into the `notebooks/` folder,
then open any of the three in Jupyter or VS Code and Run All. Each is fully
self-contained and runs independently of the others.

The Neural Network notebook's preprocessing section is worth knowing well: it's the
only one that adds `StandardScaler` to the numeric features, because unlike trees,
a neural network's weights are sensitive to feature scale.

## Running the full production training script

Needs both `train.csv` and `test.csv` in the same folder:
```
python3 train_model.py
```
Takes about 4 minutes; reproduces `final_model.pkl`, `preprocessing.pkl` and a fresh
`submission.csv`.

## Key results (for quick reference)

| Model | Accuracy | Macro F1 | ROC-AUC | Notes |
|---|---|---|---|---|
| LightGBM (production, full data) | 94.0% | 0.867 | — | Deployed in both apps |
| XGBoost (150k sample) | 96.6% | 0.906 | 0.982 | Best of the notebook comparison |
| Neural Network (150k sample) | 96.4% | 0.900 | 0.980 | Very close second — needed StandardScaler |
| Random Forest (150k sample) | 94.6% | 0.875 | 0.978 | |
