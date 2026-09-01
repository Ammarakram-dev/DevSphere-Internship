"""
DevSphere AI & ML Internship
Week 1: Python + Pandas

Project:
Student Performance Data Analysis and Cleaning

Description:
This script loads, analyzes, validates, cleans, and exports
the Student Performance dataset using Python and Pandas.
"""

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# 1. Project Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA = DATA_DIR / "student-mat.csv"
CLEANED_DATA = DATA_DIR / "student-mat-cleaned.csv"


# ---------------------------------------------------------
# 2. Load Dataset
# ---------------------------------------------------------

print("=" * 60)
print("STUDENT PERFORMANCE DATA ANALYSIS")
print("DevSphere AI & ML Internship - Week 1")
print("=" * 60)

df = pd.read_csv(RAW_DATA, sep=";")

print("\nDataset loaded successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# ---------------------------------------------------------
# 3. First Five Rows
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FIRST FIVE ROWS")
print("=" * 60)

print(df.head())


# ---------------------------------------------------------
# 4. Dataset Information
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

df.info()


# ---------------------------------------------------------
# 5. Summary Statistics
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print(df.describe())


print("\nCategorical Summary Statistics")
print(df.describe(include="object"))


# ---------------------------------------------------------
# 6. Missing Value Analysis
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing_values = df.isnull().sum()

print("Missing values by column:")
print(missing_values)

total_missing = df.isnull().sum().sum()

print(f"\nTotal missing values: {total_missing}")


# ---------------------------------------------------------
# 7. Duplicate Analysis
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE ANALYSIS")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_count}")


# ---------------------------------------------------------
# 8. Data Type Analysis
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DATA TYPE ANALYSIS")
print("=" * 60)

print(df.dtypes)

print("\nColumn counts by data type:")
print(df.dtypes.value_counts())


# ---------------------------------------------------------
# 9. Categorical Data Validation
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CATEGORICAL DATA VALIDATION")
print("=" * 60)

categorical_columns = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
]

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].unique())


# ---------------------------------------------------------
# 10. Numerical Range Validation
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("NUMERICAL RANGE VALIDATION")
print("=" * 60)

numeric_columns = df.select_dtypes(include="number").columns

numeric_range = df[numeric_columns].agg(["min", "max"]).T

print(numeric_range)

negative_counts = (df[numeric_columns] < 0).sum()

print("\nNegative values by numerical column:")
print(negative_counts)


# ---------------------------------------------------------
# 11. Create Cleaned Dataset
# ---------------------------------------------------------

cleaned_df = df.copy()

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

print(f"Original shape: {df.shape}")
print(f"Cleaning dataset shape: {cleaned_df.shape}")


# ---------------------------------------------------------
# 12. Handle Missing Values
# ---------------------------------------------------------

missing_before = cleaned_df.isnull().sum().sum()

if missing_before > 0:
    cleaned_df = cleaned_df.dropna()
    print(f"\nMissing values found and removed: {missing_before}")
else:
    print("\nNo missing values found.")
    print("No missing-value removal was necessary.")

missing_after = cleaned_df.isnull().sum().sum()

print(f"Missing values after cleaning: {missing_after}")


# ---------------------------------------------------------
# 13. Handle Duplicate Records
# ---------------------------------------------------------

duplicates_before = cleaned_df.duplicated().sum()

if duplicates_before > 0:
    cleaned_df = cleaned_df.drop_duplicates()
    print(f"\nDuplicate rows removed: {duplicates_before}")
else:
    print("\nNo duplicate records found.")
    print("No duplicate rows were removed.")

duplicates_after = cleaned_df.duplicated().sum()

print(f"Duplicate rows after cleaning: {duplicates_after}")


# ---------------------------------------------------------
# 14. Final Data Quality Check
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL DATA QUALITY CHECK")
print("=" * 60)

print(f"Rows: {cleaned_df.shape[0]}")
print(f"Columns: {cleaned_df.shape[1]}")
print(f"Missing values: {cleaned_df.isnull().sum().sum()}")
print(f"Duplicate rows: {cleaned_df.duplicated().sum()}")


# ---------------------------------------------------------
# 15. Export Cleaned Dataset
# ---------------------------------------------------------

cleaned_df.to_csv(CLEANED_DATA, index=False)

print("\nCleaned dataset exported successfully.")
print(f"Saved to: {CLEANED_DATA}")


# ---------------------------------------------------------
# 16. Verify Exported Dataset
# ---------------------------------------------------------

verified_df = pd.read_csv(CLEANED_DATA)

print("\n" + "=" * 60)
print("EXPORTED DATASET VERIFICATION")
print("=" * 60)

print(f"Shape: {verified_df.shape}")
print(f"Missing values: {verified_df.isnull().sum().sum()}")
print(f"Duplicate rows: {verified_df.duplicated().sum()}")


# ---------------------------------------------------------
# 17. Basic Grade Analysis
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("BASIC GRADE ANALYSIS")
print("=" * 60)

grade_columns = ["G1", "G2", "G3"]

average_grades = cleaned_df[grade_columns].mean()

print("Average student grades:")

for grade, average in average_grades.items():
    print(f"{grade}: {average:.2f}")


# ---------------------------------------------------------
# 18. Final Grade Analysis
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL GRADE ANALYSIS")
print("=" * 60)

print(f"Average: {cleaned_df['G3'].mean():.2f}")
print(f"Minimum: {cleaned_df['G3'].min()}")
print(f"Maximum: {cleaned_df['G3'].max()}")
print(f"Median: {cleaned_df['G3'].median()}")


# ---------------------------------------------------------
# 19. Gender-Based Grade Analysis
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("AVERAGE FINAL GRADE BY GENDER")
print("=" * 60)

gender_analysis = (
    cleaned_df.groupby("sex")["G3"]
    .mean()
    .round(2)
    .sort_values(ascending=False)
)

print(gender_analysis)


# ---------------------------------------------------------
# 20. Completion Message
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)

print("The dataset was loaded, analyzed, validated, cleaned,")
print("and exported successfully.")