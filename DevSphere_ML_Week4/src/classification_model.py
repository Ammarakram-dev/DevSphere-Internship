from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "classification_model.pkl"
SCREENSHOT_PATH = BASE_DIR / "screenshots" / "01_confusion_matrix.png"


# Load dataset
dataset = load_breast_cancer()

X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target, name="target")


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train classification model
model = LogisticRegression(
    max_iter=5000,
    random_state=42,
)

model.fit(X_train_scaled, y_train)


# Predictions
y_pred = model.predict(X_test_scaled)


# Evaluation
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("=" * 60)
print("DEVSPHERE AI & ML INTERNSHIP - WEEK 4")
print("BREAST CANCER CLASSIFICATION")
print("=" * 60)

print(f"\nDataset samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print("\nSample predictions:")
print(y_pred[:10])

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=dataset.target_names,
    )
)

print("Confusion Matrix:")
print(cm)


# Save confusion matrix
SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=dataset.target_names,
)

disp.plot()

plt.title("Classification Model - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# Save trained model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "target_names": list(dataset.target_names),
    },
    MODEL_PATH,
)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Confusion matrix saved to: {SCREENSHOT_PATH}")
print("\nWeek 4 classification model completed successfully.")