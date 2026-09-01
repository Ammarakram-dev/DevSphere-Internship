# Student Performance Data Analysis & Cleaning

### DevSphere AI & ML Internship — Week 01

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
</p>

<p align="center">
  <b>Raw Data → Validation → Cleaning → Analysis → Visualization</b>
</p>

---

## Overview

A practical Python data-analysis project focused on inspecting, validating, cleaning, and analyzing the **Student Performance — Mathematics** dataset.

The project follows a structured data workflow while preserving the original dataset and producing a separately validated cleaned dataset.

---

## Dataset

| Property    |        Value |
| ----------- | -----------: |
| Records     |      **395** |
| Columns     |       **33** |
| Subject     |  Mathematics |
| Main Grades | G1 • G2 • G3 |
| Format      |          CSV |
| Separator   |          `;` |

### Academic Variables

- **G1** — First period grade
- **G2** — Second period grade
- **G3** — Final grade

---

## Workflow

```text
┌──────────────┐
│ Load Dataset │
└──────┬───────┘
       ↓
┌────────────────┐
│ Initial Review │
└──────┬─────────┘
       ↓
┌──────────────────┐
│ Data Quality     │
│ Checks           │
└──────┬───────────┘
       ↓
┌──────────────────┐
│ Create Clean     │
│ Dataset Copy     │
└──────┬───────────┘
       ↓
┌──────────────────┐
│ Validate &       │
│ Export           │
└──────┬───────────┘
       ↓
┌──────────────────┐
│ Grade Analysis   │
└──────┬───────────┘
       ↓
┌──────────────────┐
│ Visualization    │
└──────────────────┘
```

---

## Analysis Performed

### Dataset Inspection

- Loaded the CSV dataset with Pandas
- Reviewed the first five records
- Checked dataset dimensions
- Inspected columns and data types
- Generated numerical statistics
- Reviewed categorical information

### Data Quality

- Checked missing values
- Checked duplicate records
- Inspected categorical values
- Reviewed numerical ranges
- Validated grade columns
- Preserved the original raw dataset

### Cleaning

A separate copy was created before cleaning:

```python
cleaned_df = df.copy()
```

The dataset contained **0 missing values** and **0 duplicate records**, so no valid records were unnecessarily removed.

### Export

The validated dataset was exported as:

```text
student-mat-cleaned.csv
```

The exported file was then loaded again to verify its integrity.

---

## Grade Analysis

The following academic variables were analyzed:

```text
G1  →  First Period Grade
G2  →  Second Period Grade
G3  →  Final Grade
```

Basic statistics were calculated for the final grade, including:

- Mean
- Median
- Minimum
- Maximum

The average final grade was also compared across gender groups using Pandas `groupby()`.

---

## Visualization

A histogram was created with Matplotlib to visualize the distribution of final mathematics grades.

```python
plt.figure(figsize=(8, 5))

plt.hist(cleaned_df["G3"], bins=10)

plt.title("Distribution of Final Mathematics Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")

plt.show()
```

---

## Project Evidence

The `screenshots` directory contains visual evidence of the main stages of the analysis.

| #   | Evidence                           |
| --- | ---------------------------------- |
| 01  | Dataset loading and preview        |
| 02  | Dataset information and statistics |
| 03  | Data quality analysis              |
| 04  | Data cleaning and validation       |
| 05  | Export and verification            |
| 06  | Final grade visualization          |

---

## Project Structure

```text
DevSphere_ML_Week1/
│
├── data/
│   ├── student-mat.csv
│   └── student-mat-cleaned.csv
│
├── notebook/
│   └── DevSphere_ML_Week1_Analysis.ipynb
│
├── screenshots/
│   ├── 01_load_and_preview.png
│   ├── 02_dataset_info_statistics.png
│   ├── 03_data_quality_analysis.png
│   ├── 04_data_cleaning.png
│   ├── 05_export_verification.png
│   └── 06_analysis_visualization.png
│
├── report/
│   ├── DevSphere_ML_Week1_Report.docx
│   └── DevSphere_ML_Week1_Report.pdf
│
├── src/
│   └── data_analysis.py
│
└── README.md
```

---

## Key Result

The dataset was successfully:

**Loaded → Inspected → Validated → Cleaned → Exported → Verified → Analyzed → Visualized**

The final cleaned dataset preserves the complete set of **395 student records and 33 columns**, with no missing values or duplicate records identified during validation.

---

## Technologies

<p align="center">
  <code>Python</code>
  <code>Pandas</code>
  <code>Matplotlib</code>
  <code>Jupyter Notebook</code>
  <code>Visual Studio Code</code>
</p>

---

<p align="center">
  <b>DEVSPHERE AI & ML INTERNSHIP</b><br>
  Week 01 • Student Performance Data Analysis & Cleaning
</p>
