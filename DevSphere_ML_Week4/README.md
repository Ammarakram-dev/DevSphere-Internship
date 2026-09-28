<div align="center">

# DevSphere AI & ML Internship — Week 4

### Classification Models

**Breast Cancer Classification using Logistic Regression**

<br>

<img src="https://img.shields.io/badge/Machine%20Learning-Classification-111827?style=for-the-badge&logo=scikitlearn&logoColor=white" />
<img src="https://img.shields.io/badge/Python-3.x-111827?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Scikit--learn-Model-111827?style=for-the-badge&logo=scikitlearn&logoColor=white" />
<img src="https://img.shields.io/badge/Status-Completed-111827?style=for-the-badge&logo=checkmarx&logoColor=white" />

</div>

---

## ✦ Project

A supervised machine learning classification project developed for **Week 4 of the DevSphere AI & ML Internship**.

The project trains a **Logistic Regression** model to classify breast cancer samples into two classes:

> **Malignant** · **Benign**

The complete workflow covers dataset preparation, train/test splitting, feature scaling, model training, prediction, evaluation, visualization, and model saving.

---

## ⚡ Workflow

```text
┌──────────────────┐
│  Load Dataset    │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Prepare Features │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Train/Test Split │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Feature Scaling  │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Train Model      │
│ Logistic Reg.    │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Generate         │
│ Predictions      │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Evaluate Model   │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Save Trained     │
│ Model            │
└──────────────────┘