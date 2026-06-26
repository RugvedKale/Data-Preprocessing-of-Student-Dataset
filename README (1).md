# Student Dataset — Data Preprocessing

An end-to-end data preprocessing pipeline applied to a student performance dataset using Python and Pandas. Completed as **TASK-01**, this project covers everything needed to take a raw, messy dataset to a clean, fully numerical one ready for machine learning model training.

## Overview

The dataset records student demographics and academic behavior, with the goal of eventually predicting `Final_Score`. Before any modeling can happen, the raw data needs to be explored, cleaned of missing values, and converted into a fully numerical format — which is exactly what this notebook does.

## Dataset

- **Source:** `student_data_100_preprocess.csv` — 100 rows, 6 columns
- **Columns:**
  - `Gender` — categorical (Male/Female)
  - `Age` — numerical
  - `Hours_Studied` — numerical
  - `Attendance` — numerical
  - `Parental_Education` — categorical (Graduate, High School, Postgraduate)
  - `Final_Score` — numerical (target variable)
- **Missing values (before cleaning):** 47 total, spread across 5 of the 6 columns

| Column              | Missing Values |
|----------------------|:--------------:|
| Age                  | 9              |
| Hours_Studied        | 10             |
| Attendance           | 8              |
| Parental_Education   | 13             |
| Final_Score          | 7              |
| Gender               | 0              |

## Workflow

1. **Load & explore** — read the CSV, inspect shape, column names, data types, and a statistical summary via `.describe()`.
2. **Identify column types** — split columns into numerical (`Age`, `Hours_Studied`, `Attendance`, `Final_Score`) and categorical (`Gender`, `Parental_Education`).
3. **Detect missing values** — used `isnull().sum()` to count missing entries per column and in total.
4. **Handle missing values**:
   - Numerical columns filled with their **mean**
   - `Parental_Education` filled with its **mode** (`Graduate`)
5. **Verify cleaning** — re-ran the missing-value check and `.info()` to confirm zero nulls remain.
6. **Encode categorical variables**:
   - `Gender` → **Label Encoding**
   - `Parental_Education` → **One-Hot Encoding** (via `pd.get_dummies`)
7. **Confirm full numerical conversion** — re-ran `.info()` to verify no `object`-type columns remain.
8. **Prepare for modeling** — separated features (`X`) from the target (`Final_Score`), then split into an 80/20 train-test set with `random_state=42`.

## Results

**Imputation values used:**

| Column         | Fill Strategy | Value Used |
|----------------|---------------|:----------:|
| Age            | Mean          | 19.42      |
| Hours_Studied  | Mean          | 4.37       |
| Attendance     | Mean          | 76.68      |
| Final_Score    | Mean          | 74.25      |
| Parental_Education | Mode      | Graduate   |

**Final dataset shape after encoding:** 100 rows × 8 columns (`Gender`, `Age`, `Hours_Studied`, `Attendance`, `Final_Score`, plus 3 one-hot columns for `Parental_Education`)

**Train/test split:**

| Set          | Shape   |
|--------------|:-------:|
| X_train      | (80, 7) |
| X_test       | (20, 7) |
| y_train      | (80,)   |
| y_test       | (20,)   |

## Key Takeaways

- **Mean imputation** is a simple, reasonable default for numerical columns with a moderate amount of missing data, preserving the overall distribution without dropping rows.
- **Mode imputation** is the natural choice for a categorical column like `Parental_Education`, filling gaps with the most common category.
- **Label Encoding vs. One-Hot Encoding** were applied deliberately based on the nature of each categorical column — `Gender` is binary, so Label Encoding is sufficient, while `Parental_Education` has more than two unordered categories, making One-Hot Encoding the safer choice to avoid implying a false ordinal relationship.
- After preprocessing, the dataset is fully numerical and split into train/test sets, ready to be handed off to any regression model to predict `Final_Score`.

## Repository Contents

```
.
├── student_data_100_preprocess.csv   # Raw dataset
├── student_dataset_task.ipynb        # Full preprocessing notebook
└── README.md
```

## Tech Stack

- Python
- pandas
- scikit-learn (`LabelEncoder`, `train_test_split`)

## Running the Notebook

```bash
pip install pandas scikit-learn
jupyter notebook student_dataset_task.ipynb
```

Make sure `student_data_100_preprocess.csv` is in the same directory as the notebook before running.
