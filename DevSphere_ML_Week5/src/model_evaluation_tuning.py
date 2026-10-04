from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "optimized_classification_model.pkl"
COMPARISON_PATH = BASE_DIR / "screenshots" / "01_model_comparison.png"
TUNING_PATH = BASE_DIR / "screenshots" / "02_tuning_results.png"
CONFUSION_PATH = BASE_DIR / "screenshots" / "03_confusion_matrix.png"


# ============================================================
# 1. LOAD DATASET
# ============================================================

dataset = load_breast_cancer()

X = pd.DataFrame(
    dataset.data,
    columns=dataset.feature_names
)

y = pd.Series(
    dataset.target,
    name="target"
)

print("=" * 70)
print("DEVSPHERE AI & ML INTERNSHIP — WEEK 5")
print("MODEL EVALUATION + TUNING")
print("=" * 70)

print(f"\nDataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print(f"Classes: {list(dataset.target_names)}")


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# 3. DEFINE MULTIPLE MODELS
# ============================================================

models = {
    "Logistic Regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    random_state=42
                ),
            ),
        ]
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    ),

    "SVM": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(kernel="rbf")
            ),
        ]
    ),
}


# ============================================================
# 4. TRAIN AND COMPARE MODELS
# ============================================================

results = []

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
        }
    )

    print(f"\n{name}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")


results_df = pd.DataFrame(results)

print("\n")
print(results_df.to_string(index=False))


# ============================================================
# 5. MODEL COMPARISON CHART
# ============================================================

COMPARISON_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

ax = results_df.set_index("Model")[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ]
].plot(
    kind="bar",
    figsize=(11, 6),
)

ax.set_title(
    "Machine Learning Model Performance Comparison"
)

ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.set_ylim(0.80, 1.00)

ax.grid(
    axis="y",
    alpha=0.25
)

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    COMPARISON_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. HYPERPARAMETER TUNING — RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING — RANDOM FOREST")
print("=" * 70)

parameter_grid = {
    "n_estimators": [
        100,
        200,
        300
    ],

    "max_depth": [
        None,
        5,
        10
    ],

    "min_samples_split": [
        2,
        5
    ],
}


grid_search = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=42
    ),

    param_grid=parameter_grid,

    cv=5,

    scoring="accuracy",

    n_jobs=-1,
)


grid_search.fit(
    X_train,
    y_train
)


best_model = grid_search.best_estimator_


print("\nBest Parameters:")

print(
    grid_search.best_params_
)


print("\nBest Cross-Validation Accuracy:")

print(
    f"{grid_search.best_score_:.4f}"
)


# ============================================================
# 7. OPTIMIZED MODEL EVALUATION
# ============================================================

optimized_predictions = best_model.predict(
    X_test
)


optimized_accuracy = accuracy_score(
    y_test,
    optimized_predictions
)


optimized_precision = precision_score(
    y_test,
    optimized_predictions
)


optimized_recall = recall_score(
    y_test,
    optimized_predictions
)


optimized_f1 = f1_score(
    y_test,
    optimized_predictions
)


print("\n" + "=" * 70)
print("OPTIMIZED MODEL PERFORMANCE")
print("=" * 70)


print(
    f"\nAccuracy : {optimized_accuracy:.4f}"
)

print(
    f"Precision: {optimized_precision:.4f}"
)

print(
    f"Recall   : {optimized_recall:.4f}"
)

print(
    f"F1-Score : {optimized_f1:.4f}"
)


print("\nClassification Report:")


print(
    classification_report(
        y_test,
        optimized_predictions,
        target_names=dataset.target_names,
    )
)


# ============================================================
# 8. TUNING RESULTS CHART
# ============================================================

cv_results = pd.DataFrame(
    grid_search.cv_results_
)


top_results = (
    cv_results[
        [
            "param_n_estimators",
            "param_max_depth",
            "param_min_samples_split",
            "mean_test_score",
        ]
    ]
    .sort_values(
        "mean_test_score",
        ascending=False
    )
    .head(10)
)


tuning_labels = []


for _, row in top_results.iterrows():

    depth = row["param_max_depth"]

    if pd.isna(depth):
        depth = "None"

    tuning_labels.append(
        f"Trees={row['param_n_estimators']}, "
        f"Depth={depth}, "
        f"Split={row['param_min_samples_split']}"
    )


plt.figure(
    figsize=(12, 6)
)


plt.barh(
    range(len(top_results)),
    top_results["mean_test_score"]
)


plt.yticks(
    range(len(top_results)),
    tuning_labels
)


plt.xlabel(
    "Cross-Validation Accuracy"
)


plt.title(
    "Top Hyperparameter Configurations"
)


plt.xlim(
    0.90,
    1.00
)


plt.gca().invert_yaxis()


plt.tight_layout()


plt.savefig(
    TUNING_PATH,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    optimized_predictions
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=dataset.target_names
)


disp.plot()


plt.title(
    "Optimized Random Forest — Confusion Matrix"
)


plt.tight_layout()


plt.savefig(
    CONFUSION_PATH,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 10. SAVE OPTIMIZED MODEL
# ============================================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


joblib.dump(
    {
        "model": best_model,
        "feature_names": list(X.columns),
        "target_names": list(dataset.target_names),
        "best_parameters": grid_search.best_params_,
        "test_accuracy": optimized_accuracy,
        "test_precision": optimized_precision,
        "test_recall": optimized_recall,
        "test_f1_score": optimized_f1,
    },
    MODEL_PATH
)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)


print(
    "\nOptimized Model: Random Forest"
)


print(
    f"Test Accuracy: {optimized_accuracy:.4%}"
)


print(
    f"Test Precision: {optimized_precision:.4%}"
)


print(
    f"Test Recall: {optimized_recall:.4%}"
)


print(
    f"Test F1-Score: {optimized_f1:.4%}"
)


print(
    "\nModel saved to:"
)


print(
    MODEL_PATH
)


print(
    "\nCharts saved:"
)


print(
    COMPARISON_PATH
)


print(
    TUNING_PATH
)


print(
    CONFUSION_PATH
)


print(
    "\nWeek 5 model evaluation and tuning completed successfully."
)