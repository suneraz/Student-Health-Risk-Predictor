"""
Example: how to load and use any of the three saved comparison models.

Each .pkl file is self-contained: it bundles the fitted preprocessing
(OneHotEncoder + numeric handling) together with the trained model in a
single scikit-learn Pipeline, so you don't need to redo any preprocessing
by hand.

Usage:
    python3 load_and_predict_example.py
"""
import joblib
import pandas as pd

# Change this to any of: neural_network_model.pkl, random_forest_model.pkl, xgboost_model.pkl
MODEL_FILE = "xgboost_model.pkl"

bundle = joblib.load(MODEL_FILE)
pipeline = bundle["pipeline"]
feature_columns = bundle["feature_columns"]

print(f"Loaded model: {bundle['model_name']}")
print(f"Test accuracy: {bundle['test_accuracy']:.4f}")
print(f"Test macro-F1: {bundle['test_macro_f1']:.4f}")
print(f"Classes: {bundle['target_classes']}")

# One example student
student = pd.DataFrame([{
    "sleep_duration": 7.5,
    "heart_rate": 72,
    "bmi": 22.0,
    "calorie_expenditure": 2400,
    "step_count": 9000,
    "exercise_duration": 40,
    "water_intake": 2.5,
    "diet_type": "balanced",
    "stress_level": "low",
    "sleep_quality": "good",
    "physical_activity_level": "active",
    "smoking_alcohol": "no",
    "gender": "male",
}])

prediction = pipeline.predict(student[feature_columns])

# All three models here (Neural Network, XGBoost, and also Random Forest via a
# shared target_encoder) were trained on a label-encoded target, so the raw
# prediction is a number (0/1/2) that needs mapping back to a text label.
if "target_encoder" in bundle:
    prediction = bundle["target_encoder"].inverse_transform(prediction)

probabilities = pipeline.predict_proba(student[feature_columns])[0]

print(f"\nPrediction: {prediction[0]}")
print("Class probabilities:")
for cls, prob in zip(bundle["target_classes"], probabilities):
    print(f"  {cls}: {prob:.3f}")
