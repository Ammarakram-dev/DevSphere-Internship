"""
DevSphere AI & ML Internship — Week 3
Supervised Learning — Regression

Project:
Student Final Grade Prediction using Linear Regression
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "student-mat.csv"
MODEL_PATH = BASE_DIR / "model" / "linear_regression_student_grade_model.pkl"


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH, sep=";")

print("=" * 60)
print("DEVSPHERE AI & ML INTERNSHIP — WEEK 3")
print("STUDENT FINAL GRADE PREDICTION")
print("=" * 60)

print("\nDataset shape:", df.shape)


# ---------------------------------------------------------
# Features and target
# ---------------------------------------------------------

X = df[["G1", "G2"]]
y = df["G3"]

print("\nFeatures:", list(X.columns))
print("Target: G3")


# ---------------------------------------------------------
# Train/Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ---------------------------------------------------------
# Train Linear Regression
# ---------------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)


# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n" + "-" * 60)
print("MODEL RESULTS")
print("-" * 60)

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

print("\nLearned Regression Equation:")
print(
    f"G3 = {model.intercept_:.4f}"
    f" + ({model.coef_[0]:.4f} × G1)"
    f" + ({model.coef_[1]:.4f} × G2)"
)


# ---------------------------------------------------------
# Prediction sample
# ---------------------------------------------------------

prediction_results = pd.DataFrame({
    "Actual G3": y_test.values,
    "Predicted G3": np.round(y_pred, 2),
})

print("\nPrediction Sample:")
print(prediction_results.head(10).to_string(index=False))


# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print("\n" + "-" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("-" * 60)
print(MODEL_PATH)

print("\nWeek 3 regression pipeline completed successfully.")