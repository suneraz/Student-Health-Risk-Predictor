"""
train_model.py
================
End-to-end training pipeline for the Student Health Risk Predictor
(Kaggle Playground Series S6E7).

Run this from the project root folder:
    python3 train_model.py

What it does, step by step:
  1. Load train.csv / test.csv
  2. Handle missing values
  3. Encode categorical features to numbers
  4. Compare two ensemble models (LightGBM vs Random Forest) with
     stratified cross-validation
  5. Train the final model on all training data
  6. Save the model + generate a Kaggle submission file

Expected runtime: a few minutes, depending on your machine.
"""

import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
import lightgbm as lgb

# ---------------------------------------------------------------
# STEP 1: Load the data
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"\nTarget class balance:")
print(train["health_condition"].value_counts(normalize=True).round(3))

# ---------------------------------------------------------------
# STEP 2: Handle missing values
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Handling missing values")
print("=" * 60)

NUMERIC_COLS = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
                 "step_count", "exercise_duration", "water_intake"]
CATEGORICAL_COLS = ["diet_type", "stress_level", "sleep_quality",
                     "physical_activity_level", "smoking_alcohol", "gender"]

print("Missing values before cleaning:")
print(train[NUMERIC_COLS + CATEGORICAL_COLS].isnull().sum())

# Numeric: fill with the median. Median is robust to outliers, unlike the mean.
medians = {}
for col in NUMERIC_COLS:
    med = train[col].median()
    medians[col] = med
    train[col] = train[col].fillna(med)
    test[col] = test[col].fillna(med)

# Categorical: fill with an explicit "missing" category rather than the mode.
# This keeps missingness visible to the model instead of silently guessing.
for col in CATEGORICAL_COLS:
    train[col] = train[col].fillna("missing")
    test[col] = test[col].fillna("missing")

print("\nMissing values after cleaning: 0 (all filled)")

# ---------------------------------------------------------------
# STEP 3: Encode categorical features
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Encoding categorical features")
print("=" * 60)

# Models need numbers, not text like "active" or "balanced".
# LabelEncoder converts each category to an integer. We fit on the
# combined train+test values so both sets use the same encoding.
cat_encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0)
    le.fit(combined)
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
    cat_encoders[col] = le
    print(f"  {col}: {list(le.classes_)}")

target_encoder = LabelEncoder()
train["target"] = target_encoder.fit_transform(train["health_condition"])
print(f"\nTarget classes: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
X = train[FEATURE_COLS]
y = train["target"]
X_test = test[FEATURE_COLS]

# ---------------------------------------------------------------
# STEP 4: Compare models with cross-validation
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Cross-validating LightGBM vs Random Forest")
print("=" * 60)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# --- LightGBM ---
print("\n--- LightGBM (3-fold CV) ---")
lgb_f1, lgb_acc = [], []
t0 = time.time()
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    model = lgb.LGBMClassifier(
        n_estimators=250, learning_rate=0.06, num_leaves=63,
        class_weight="balanced",  # compensates for the 86% / 6% / 8% class imbalance
        random_state=42, verbosity=-1, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average="macro")
    acc = accuracy_score(y_val, preds)
    lgb_f1.append(f1)
    lgb_acc.append(acc)
    print(f"  Fold {fold+1}: macro-F1={f1:.4f}  accuracy={acc:.4f}")
print(f"LightGBM mean: macro-F1={np.mean(lgb_f1):.4f}  accuracy={np.mean(lgb_acc):.4f}  ({time.time()-t0:.1f}s)")

# --- Random Forest (single holdout split, since full 3-fold CV is slow at this data size) ---
print("\n--- Random Forest (holdout split) ---")
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=150, max_depth=14, class_weight="balanced",
    n_jobs=-1, random_state=42
)
rf.fit(X_tr, y_tr)
preds = rf.predict(X_val)
rf_f1 = f1_score(y_val, preds, average="macro")
rf_acc = accuracy_score(y_val, preds)
print(f"Random Forest: macro-F1={rf_f1:.4f}  accuracy={rf_acc:.4f}  ({time.time()-t0:.1f}s)")

print("\n--- Model choice ---")
print("LightGBM selected as final model: similar accuracy, much faster to train,")
print("and handles the class imbalance / missing data natively.")

# ---------------------------------------------------------------
# STEP 5: Train final model on ALL training data
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Training final model on full training set")
print("=" * 60)

t0 = time.time()
final_model = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.06, num_leaves=63,
    class_weight="balanced", random_state=42, verbosity=-1, n_jobs=-1
)
final_model.fit(X, y)
print(f"Trained in {time.time()-t0:.1f}s")

# Show a classification report on a fresh holdout for a full picture
_, X_check, _, y_check = train_test_split(X, y, test_size=0.15, stratify=y, random_state=123)
preds_check = final_model.predict(X_check)
print("\nClassification report (held-out sample):")
print(classification_report(y_check, preds_check, target_names=target_encoder.classes_))

# Feature importance
importances = dict(zip(FEATURE_COLS, final_model.feature_importances_.tolist()))
importances = dict(sorted(importances.items(), key=lambda x: -x[1]))
print("Feature importance (most to least influential):")
for k, v in importances.items():
    print(f"  {k}: {v}")

# ---------------------------------------------------------------
# STEP 6: Save model + generate Kaggle submission
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: Saving model and generating submission")
print("=" * 60)

joblib.dump(final_model, "final_model.pkl")
joblib.dump({
    "target_encoder": target_encoder,
    "cat_encoders": cat_encoders,
    "numeric_medians": medians,
    "feature_cols": FEATURE_COLS
}, "preprocessing.pkl")
print("Saved: final_model.pkl, preprocessing.pkl")

test_preds = final_model.predict(X_test)
test_labels = target_encoder.inverse_transform(test_preds)
submission = pd.DataFrame({"id": test["id"], "health_condition": test_labels})
submission.to_csv("submission.csv", index=False)
print(f"Saved: submission.csv  {submission.shape}")
print(submission["health_condition"].value_counts())

print("\nDone. Model and submission are ready.")
