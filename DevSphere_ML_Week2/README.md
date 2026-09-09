<div align="center">

# ✦ DEVSPHERE AI & ML INTERNSHIP

### `WEEK 02` · DATA VISUALIZATION

<img src="https://img.shields.io/badge/INTERNSHIP-DEVSPHERE-6C63FF?style=for-the-badge&labelColor=0D1117" />
<img src="https://img.shields.io/badge/PYTHON-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=0D1117" />
<img src="https://img.shields.io/badge/PANDAS-DATA%20ANALYSIS-150458?style=for-the-badge&logo=pandas&logoColor=white&labelColor=0D1117" />
<img src="https://img.shields.io/badge/MATPLOTLIB-VISUALIZATION-11557C?style=for-the-badge&labelColor=0D1117" />
<img src="https://img.shields.io/badge/SEABORN-STATISTICAL%20VIZ-4C72B0?style=for-the-badge&labelColor=0D1117" />

<br><br>

> **Transforming raw student performance data into visual stories, patterns, and insights.**

<br>

**📊 Analyze**   →   **🎨 Visualize**   →   **🧠 Interpret**   →   **💡 Understand**

</div>

---

## ⚡ Project Snapshot

<table>
<tr>
<td width="50%">

### 🎯 Objective

Build clear and meaningful visualizations from the Student Performance dataset using Python.

</td>
<td width="50%">

### 📌 Focus

Compare performance, identify trends, and understand grade distribution through visual analysis.

</td>
</tr>
</table>

---

## 🧩 What Was Built

| #    | Visualization     | Purpose                                   |
| ---- | ----------------- | ----------------------------------------- |
| `01` | 📊 **Bar Chart**  | Compare average final grade by gender     |
| `02` | 📈 **Line Chart** | Track average G1 → G2 → G3                |
| `03` | 🥧 **Pie Chart**  | Show final-grade performance distribution |

---

## 🛠️ Technology Stack

<div align="center">

|        Technology       | Role                             |
| :---------------------: | :------------------------------- |
|      🐍 **Python**      | Core programming language        |
|      🐼 **Pandas**      | Data loading & manipulation      |
|    📊 **Matplotlib**    | Data visualization               |
|      🎨 **Seaborn**     | Statistical visualization        |
| 📓 **Jupyter Notebook** | Interactive analysis environment |

</div>

---

# 📊 Visualization Gallery

## 01 · Average Final Grade by Gender

<table>
<tr>
<td width="55%">

<img src="screenshots/01_bar_chart.png" width="100%" />

</td>
<td width="45%">

### 🔍 What it shows

The bar chart compares the average **final grade (`G3`)** between female and male students.

### 💡 Insight

The averages are relatively close, suggesting that gender alone does not represent a major difference in final performance within this dataset.

</td>
</tr>
</table>

---

## 02 · Average Grades Across G1, G2 & G3

<table>
<tr>
<td width="45%">

### 🔍 What it shows

The line chart follows the average student grades across:

`G1` → `G2` → `G3`

### 💡 Insight

The visualization provides a simple view of how average academic performance changes across the three grading periods.

</td>
<td width="55%">

<img src="screenshots/02_line_chart.png" width="100%" />

</td>
</tr>
</table>

---

## 03 · Final Grade Performance Distribution

<table>
<tr>
<td width="55%">

<img src="screenshots/03_pie_chart.png" width="100%" />

</td>
<td width="45%">

### 📌 Performance Groups

**High** → `15–20`

**Medium** → `10–14`

**Low** → `0–9`

### 💡 Insight

The pie chart provides a quick visual overview of how students are distributed across different final-performance levels.

</td>
</tr>
</table>

---

# 🧠 Insight Engine

<details>
<summary><strong>📊 Bar Chart — Performance Comparison</strong></summary>

The average final grades of female and male students are relatively close. This indicates that there is no large difference in average final performance between the two groups in this dataset.

</details>

<details>
<summary><strong>📈 Line Chart — Grade Trend</strong></summary>

The G1, G2, and G3 averages provide a useful overview of academic performance across the grading periods. The line makes changes between the three values easy to recognize.

</details>

<details>
<summary><strong>🥧 Pie Chart — Performance Distribution</strong></summary>

Grouping students into high, medium, and low performance levels provides a simple way to understand the overall distribution of final grades.

</details>

---

# 📁 Project Architecture

```text
DevSphere_ML_Week2/
│
├── 📂 data/
│   └── student-mat.csv
│
├── 📂 notebook/
│   └── DevSphere_ML_Week2_Visualization.ipynb
│
├── 📂 screenshots/
│   ├── 01_bar_chart.png
│   ├── 02_line_chart.png
│   └── 03_pie_chart.png
│
├── 📂 report/
│   ├── DevSphere_ML_Week2_Report.docx
│   └── DevSphere_ML_Week2_Report.pdf
│
├── 📂 src/
│
└── 📄 README.md
```

---

# 🔬 Dataset

### Student Performance — Mathematics

The project uses the **Student Performance Mathematics dataset** containing:

```text
395 Students
33 Attributes
G1 → First-period grade
G2 → Second-period grade
G3 → Final grade
```

The dataset was loaded using Pandas and used as the foundation for the visual analysis.

---

# 🧪 Analysis Workflow

```text
                RAW DATA
                   │
                   ▼
            ┌─────────────┐
            │   Pandas    │
            │ Load Dataset│
            └──────┬──────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Data Exploration│
          └────────┬────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Visualization Layer │
        └──────────┬──────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
      📊 Bar     📈 Line     🥧 Pie
        │          │          │
        └──────────┼──────────┘
                   ▼
             🧠 INSIGHTS
                   │
                   ▼
             💡 CONCLUSION
```

---

# 🎓 Learning Outcomes

Through this task, I practiced:

* 📊 Creating meaningful data visualizations
* 🐼 Manipulating datasets with Pandas
* 🎨 Building charts with Matplotlib and Seaborn
* 🧠 Extracting insights from visual patterns
* 📓 Working with Jupyter Notebook
* 📝 Presenting analytical results clearly

---

# 📓 Notebook

The complete analysis is available here:

```text
notebook/DevSphere_ML_Week2_Visualization.ipynb
```

The notebook contains the complete visualization workflow, including dataset loading, chart creation, and interpretation.

---

# 📄 Documentation

A complete project report is included in both formats:

```text
report/DevSphere_ML_Week2_Report.pdf
report/DevSphere_ML_Week2_Report.docx
```

---

# 🚀 Final Result

This project demonstrates how raw numerical information can be transformed into understandable visual insights.

Instead of viewing grades only as rows and columns, the visualizations make it easier to **compare performance, observe trends, and understand distributions**.

<div align="center">

### `DATA → VISUALIZATION → INSIGHT`

<br>

**Week 02 — Successfully Completed**

<br>

<img src="https://img.shields.io/badge/STATUS-COMPLETE-00C853?style=for-the-badge&labelColor=0D1117" />

<br><br>

`DevSphere AI & ML Internship`

</div>

---

<div align="center">

### ✦ Built with Python · Pandas · Matplotlib · Seaborn ✦

</div>
