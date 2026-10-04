<div align="center">

# 🧬 Classification Models

### DevSphere AI & ML Internship — Week 4

*Building, evaluating, and persisting a supervised classification model.*

<br>

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Model](https://img.shields.io/badge/Model-Logistic%20Regression-0A66C2?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-98.25%25-2EA44F?style=for-the-badge)

</div>

---

## 📌 Overview

Week 4 of the **DevSphere AI & ML Internship** introduces **supervised classification**. This project builds a **Logistic Regression** classifier on the Breast Cancer Wisconsin Diagnostic dataset, evaluates it with standard classification metrics, visualizes the results, and saves the trained model for reuse.

---

## 🎯 Objective

- Understand supervised classification
- Prepare classification data
- Train a classification model
- Evaluate model performance
- Visualize classification results
- Save the trained model

---

## 🗂️ Dataset

| Property | Details |
|---|---|
| **Source** | Breast Cancer Wisconsin Diagnostic Dataset |
| **Loader** | `sklearn.datasets.load_breast_cancer()` |
| **Samples** | 569 |
| **Features** | 30 numerical features |
| **Task** | Binary classification |
| **Train / Test Split** | 80% / 20% |
| **Split Strategy** | Stratified |
| **Random State** | `42` |

---

## 🔄 Machine Learning Workflow

```text
        ┌─────────────────────┐
        │       Dataset       │
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │   Data Preparation  │
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │   Feature Scaling   │
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │ Logistic Regression │
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │   Model Evaluation  │
        └──────────┬──────────┘
                   ↓
        ┌─────────────────────┐
        │ Results & Visuals   │
        └─────────────────────┘
```

---

## 🤖 Model

**Logistic Regression** is a linear classification algorithm that estimates the probability of a sample belonging to each class. It is a widely used and interpretable baseline for binary classification.

| Component | Details |
|---|---|
| **Classifier** | Logistic Regression |
| **Feature Scaling** | `StandardScaler` |
| **Pipeline** | Scaling and classification combined in a single workflow |
| **Data Split** | 80% training / 20% testing, stratified, `random_state=42` |

---

## 📊 Performance

<div align="center">

| Metric | Result |
|:---:|:---:|
| **Accuracy** | **98.25%** |

</div>

The model achieved **98.25% accuracy** on the held-out test set.

---

## 🧪 Evaluation

The model was evaluated using the following metrics and tools:

| Evaluation Method | Purpose |
|---|---|
| **Accuracy** | Overall proportion of correct predictions |
| **Precision** | Correctness of positive predictions |
| **Recall** | Coverage of actual positive samples |
| **F1-Score** | Balance between precision and recall |
| **Confusion Matrix** | Breakdown of correct and incorrect predictions per class |
| **Classification Report** | Per-class summary of precision, recall, and F1-score |

---

## 🖼️ Visual Results

<p align="center">
  <img src="./screenshots/classification_results.png" alt="Classification Results" width="800">
</p>

---

## 📁 Project Structure

```text
DevSphere_ML_Week4/
├── data/
├── model/
│   └── classification_model.pkl
├── notebook/
│   └── DevSphere_ML_Week4_Classification.ipynb
├── report/
│   └── DevSphere_ML_Week4_Classification_Report.docx
├── screenshots/
│   └── classification_results.png
└── src/
    └── classification_model.py
```

---

## 🛠️ Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Seaborn-4C8CBF?style=for-the-badge" alt="Seaborn">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
</p>

---

## ▶️ Run the Project

From the project root directory, run:

```bash
py src/classification_model.py
```

---

## ✅ Deliverables

- [x] Classification model
- [x] Evaluation metrics
- [x] Confusion matrix
- [x] Classification report
- [x] Visualization
- [x] Jupyter Notebook
- [x] Evaluation report
- [x] Saved model

---

## 🏁 Conclusion

The classification workflow was successfully implemented and evaluated, achieving **98.25% accuracy** with a Logistic Regression model.

<div align="center">

**DevSphere AI & ML Internship · Week 4**

</div>