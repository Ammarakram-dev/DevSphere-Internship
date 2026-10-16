
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
CHART_DIR = ROOT / "screenshots"
REPORT_DIR = ROOT / "report"

for directory in (MODEL_DIR, CHART_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# 1. Load and prepare the dataset
dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target, name="target")

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("=" * 65)
print("DEVSPHERE ML WEEK 6 | PREDICTION SYSTEM")
print("=" * 65)
print(f"Dataset samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# 2. Train and compare multiple models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=42)),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
}

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    trained_models[name] = model
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    })

# 3. Tune Logistic Regression using training data only
tuning_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000, random_state=42)),
])

param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__solver": ["liblinear", "lbfgs"],
}

search = GridSearchCV(
    tuning_pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
search.fit(X_train, y_train)

best_model = search.best_estimator_
best_predictions = best_model.predict(X_test)
best_probabilities = best_model.predict_proba(X_test)[:, 1]

results.append({
    "Model": "Tuned Logistic Regression",
    "Accuracy": accuracy_score(y_test, best_predictions),
    "Precision": precision_score(y_test, best_predictions, zero_division=0),
    "Recall": recall_score(y_test, best_predictions, zero_division=0),
    "F1-Score": f1_score(y_test, best_predictions, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, best_probabilities),
})

results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

# 4. Save the selected model and feature metadata
joblib.dump(
    {
        "model": best_model,
        "feature_names": list(dataset.feature_names),
        "target_names": list(dataset.target_names),
    },
    MODEL_DIR / "prediction_model.joblib",
)

# 5. Create performance charts
sns.set_theme(style="whitegrid")

ax = results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
].plot(kind="bar", figsize=(12, 6), ylim=(0.80, 1.01))

ax.set_title("Week 6 — Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(CHART_DIR / "01_model_comparison.png", dpi=180)
plt.close()

ConfusionMatrixDisplay.from_predictions(
    y_test,
    best_predictions,
    display_labels=dataset.target_names,
    cmap="Blues",
)
plt.title("Tuned Logistic Regression — Confusion Matrix")
plt.tight_layout()
plt.savefig(CHART_DIR / "02_confusion_matrix.png", dpi=180)
plt.close()

RocCurveDisplay.from_predictions(y_test, best_probabilities)
plt.title("Tuned Logistic Regression — ROC Curve")
plt.tight_layout()
plt.savefig(CHART_DIR / "03_roc_curve.png", dpi=180)
plt.close()

# 6. Save evaluation report
report = classification_report(
    y_test,
    best_predictions,
    target_names=dataset.target_names,
    zero_division=0,
)

with (REPORT_DIR / "evaluation_report.txt").open("w", encoding="utf-8") as file:
    file.write("DEVSPHERE ML WEEK 6 — EVALUATION REPORT\n")
    file.write("=" * 45 + "\n\n")
    file.write(f"Dataset samples: {len(X)}\n")
    file.write(f"Number of features: {X.shape[1]}\n")
    file.write("Train/test split: 80/20, stratified, random_state=42\n\n")
    file.write("Best Logistic Regression parameters:\n")
    file.write(f"{search.best_params_}\n\n")
    file.write(f"Cross-validation accuracy: {search.best_score_:.4f}\n\n")
    file.write("Test metrics:\n")
    file.write(f"Accuracy: {accuracy_score(y_test, best_predictions):.4f}\n")
    file.write(f"Precision: {precision_score(y_test, best_predictions):.4f}\n")
    file.write(f"Recall: {recall_score(y_test, best_predictions):.4f}\n")
    file.write(f"F1-score: {f1_score(y_test, best_predictions):.4f}\n")
    file.write(f"ROC-AUC: {roc_auc_score(y_test, best_probabilities):.4f}\n\n")
    file.write("Classification report:\n")
    file.write(report)
    file.write(
        "\n\nEducational demonstration only. Not for clinical diagnosis.\n"
    )

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False, formatters={
    column: "{:.4f}".format
    for column in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
}))
print("\nBest tuning parameters:", search.best_params_)
print(f"Cross-validation accuracy: {search.best_score_:.4f}")
print("\nSaved model: model/prediction_model.joblib")
print("Saved charts: screenshots/")
print("Saved reports: report/")
print("\nTraining and evaluation completed successfully.")
