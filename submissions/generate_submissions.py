"""
Generates a Kaggle submission.csv for each of the three trained models
(Neural Network, Random Forest, XGBoost), saving them all into a
"submissions" folder created next to this script.

Needs in the same folder:
    train.csv
    test.csv
    neural_network_model.pkl
    random_forest_model.pkl
    xgboost_model.pkl

Run with:
    python3 generate_submissions.py
"""
import pandas as pd
import joblib
import os

os.makedirs("submissions", exist_ok=True)

NUMERIC_FEATURES = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
                     "step_count", "exercise_duration", "water_intake"]
CATEGORICAL_FEATURES = ["diet_type", "stress_level", "sleep_quality",
                         "physical_activity_level", "smoking_alcohol", "gender"]

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print("Test shape:", test.shape)

# Same imputation used during training: numeric medians come from train.csv,
# categorical blanks become an explicit "missing" category. Needed here because
# the Neural Network can't handle raw NaN values the way the tree models can.
medians = {col: train[col].median() for col in NUMERIC_FEATURES}
for col in NUMERIC_FEATURES:
    test[col] = test[col].fillna(medians[col])
for col in CATEGORICAL_FEATURES:
    test[col] = test[col].fillna("missing")

models = [
    ("Neural Network", "neural_network_model.pkl", "submissions/submission_neural_network.csv"),
    ("Random Forest", "random_forest_model.pkl", "submissions/submission_random_forest.csv"),
    ("XGBoost", "xgboost_model.pkl", "submissions/submission_xgboost.csv"),
]

for name, fname, outpath in models:
    bundle = joblib.load(fname)
    pipeline = bundle["pipeline"]
    feature_cols = bundle["feature_columns"]

    X_test = test[feature_cols]
    pred = pipeline.predict(X_test)

    if "target_encoder" in bundle:
        pred_labels = bundle["target_encoder"].inverse_transform(pred)
    else:
        pred_labels = pred

    submission = pd.DataFrame({"id": test["id"], "health_condition": pred_labels})
    submission.to_csv(outpath, index=False)
    print(f"\n{name}: saved {outpath}")
    print(submission["health_condition"].value_counts())

print("\ndone - check the submissions folder")
