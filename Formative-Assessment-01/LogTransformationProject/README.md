# 🔬 Log Transformation & Scaling

## Regression Analysis Project

> A focused data science project that studies how **log transformation**
> and **feature scaling** affect data distribution and Linear Regression
> performance.

------------------------------------------------------------------------

# 🎯 1. Project Objective

This project compares a regression model trained on the **original
dataset** with a regression model trained after applying:

1.  **Log Transformation**
2.  **StandardScaler**
3.  **Linear Regression**
4.  **Performance Comparison**

The objective is to determine, using measured results, whether
preprocessing changes the distribution of skewed data and whether it
improves regression performance.

------------------------------------------------------------------------

# 📁 2. Project Structure

``` text
LogTransformationProject/
│
├── data/                       # Raw dataset
│
├── notebooks/                  # Jupyter Notebook analysis
│
├── outputs/                    # Generated graphs and results
│
├── requirements.txt            # Python dependencies
│
└── README.md                   # Project documentation
```

## 🏗️ Architecture

``` text
                    ┌─────────────────────┐
                    │     RAW DATASET     │
                    │       CSV File      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  DATA UNDERSTANDING │
                    │ Shape • Types       │
                    │ Missing Values      │
                    │ Statistics          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ SKEWNESS ANALYSIS   │
                    │ Histogram + .skew() │
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
   ┌─────────────────────┐          ┌─────────────────────┐
   │ BASELINE PIPELINE   │          │ TRANSFORMED PIPELINE│
   │ Original Data       │          │ Log Transformation  │
   └──────────┬──────────┘          └──────────┬──────────┘
              │                                 │
              ▼                                 ▼
   ┌─────────────────────┐          ┌─────────────────────┐
   │ Train/Test Split    │          │ Train/Test Split    │
   └──────────┬──────────┘          └──────────┬──────────┘
              │                                 │
              ▼                                 ▼
   ┌─────────────────────┐          ┌─────────────────────┐
   │ Linear Regression   │          │ StandardScaler      │
   └──────────┬──────────┘          └──────────┬──────────┘
              │                                 │
              ▼                                 ▼
   ┌─────────────────────┐          ┌─────────────────────┐
   │ MAE • RMSE • R²     │          │ Linear Regression   │
   └──────────┬──────────┘          └──────────┬──────────┘
              │                                 │
              │                                 ▼
              │                      ┌─────────────────────┐
              │                      │ MAE • RMSE • R²     │
              │                      └──────────┬──────────┘
              │                                 │
              └────────────────┬────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ RESULT COMPARISON   │
                    │ + FINAL CONCLUSION  │
                    └─────────────────────┘
```

------------------------------------------------------------------------

# 🧰 3. Technology Stack

  Technology            Purpose
  --------------------- ---------------------------------------------
  🐍 Python             Main programming language
  💻 VS Code            Development environment
  📓 Jupyter Notebook   Interactive data analysis
  🐼 pandas             Dataset loading and manipulation
  🔢 NumPy              Numerical operations and log transformation
  📊 Matplotlib         Histograms and visualizations
  🤖 scikit-learn       Splitting, scaling, regression and metrics

------------------------------------------------------------------------

# 📦 4. Requirements

Install dependencies from `requirements.txt`.

Required packages:

-   `pandas`
-   `numpy`
-   `matplotlib`
-   `scikit-learn`
-   `jupyter`
-   `ipykernel`

The project intentionally uses a small dependency set. No database, web
server, deep-learning framework, Docker, React, or backend API is
required.

------------------------------------------------------------------------

# ⚙️ 5. Installation

## Step 1 --- Install Python

Install Python 3 and ensure it is added to the system PATH.

## Step 2 --- Open the Project in VS Code

Open the `LogTransformationProject` folder.

Recommended extensions:

-   Python
-   Jupyter

## Step 3 --- Create Virtual Environment

### Windows PowerShell

``` powershell
python -m venv .venv
```

Activate:

``` powershell
.\.venv\Scripts\Activate.ps1
```

### Command Prompt

``` cmd
.venv\Scriptsctivate
```

## Step 4 --- Install Dependencies

``` powershell
pip install -r requirements.txt
```

## Step 5 --- Select Python Interpreter

In VS Code:

1.  Press `Ctrl + Shift + P`
2.  Select `Python: Select Interpreter`
3.  Choose `.venv`

## Step 6 --- Open Notebook

Open:

``` text
notebooks/analysis.ipynb
```

Select the `.venv` Python kernel.

------------------------------------------------------------------------

# 🗂️ 6. Dataset

## Recommended Dataset

Use the **Ames Housing / House Prices** dataset.

Regression target:

``` text
SalePrice
```

Place the downloaded CSV inside:

``` text
data/
```

Example candidate variables:

``` text
GrLivArea
TotalBsmtSF
1stFlrSF
GarageArea
OverallQual
SalePrice
```

The final variable selected for log transformation must be chosen after
measuring actual skewness.

------------------------------------------------------------------------

# 🔎 7. Data Understanding Pipeline

``` text
CSV DATASET
     │
     ▼
LOAD WITH PANDAS
     │
     ▼
CHECK:
• Rows and columns
• Column names
• Data types
• Missing values
• Descriptive statistics
     │
     ▼
SELECT NUMERICAL VARIABLES
     │
     ▼
CALCULATE SKEWNESS
     │
     ▼
VISUALIZE DISTRIBUTION
```

The analysis starts with evidence. A variable is selected for
transformation based on its measured distribution and skewness.

------------------------------------------------------------------------

# 📈 8. Why Log Transformation?

Log transformation is commonly applied to highly skewed positive
numerical data.

``` text
Original Values
      │
      ▼
Large values have a wide spread
      │
      ▼
Log Transformation
      │
      ▼
Large values are compressed
      │
      ▼
Distribution may become less skewed
```

The project will generally use:

``` python
np.log1p(x)
```

This calculates:

``` text
log(1 + x)
```

`np.log1p()` can handle zero values.

## Important Constraint

Do not blindly apply log transformation to invalid negative values.
Inspect the data first.

## Scientific Interpretation

Log transformation does **not** guarantee improved model accuracy. The
project compares actual metrics before and after preprocessing.

------------------------------------------------------------------------

# 📏 9. Why StandardScaler?

`StandardScaler` standardizes numerical features.

General formula:

``` text
z = (x - mean) / standard deviation
```

## Correct Procedure: Prevent Data Leakage

``` text
TRAINING DATA
     │
     ▼
Fit StandardScaler
     │
     ▼
Transform Training Data
     │
     └──────────────┐
                    │
TEST DATA           │
     │              │
     ▼              ▼
Use the SAME fitted scaler
     │
     ▼
Transform Test Data
```

Never fit the scaler independently on the test set.

------------------------------------------------------------------------

# 🧪 10. Experimental Approach

## Experiment A --- Before Transformation

``` text
Original Data
     │
     ▼
Select Features X and Target y
     │
     ▼
Train/Test Split
     │
     ▼
Linear Regression
     │
     ▼
Predictions
     │
     ▼
MAE + RMSE + R²
```

## Experiment B --- After Transformation and Scaling

``` text
Skewed Numerical Data
     │
     ▼
Log Transformation
     │
     ▼
Select Features X and Target y
     │
     ▼
Train/Test Split
     │
     ▼
Fit StandardScaler on Training Data
     │
     ▼
Transform Training and Test Data
     │
     ▼
Linear Regression
     │
     ▼
Predictions
     │
     ▼
MAE + RMSE + R²
```

------------------------------------------------------------------------

# 🔁 11. Complete Workflow

``` text
START
  │
  ▼
1. Load Dataset
  │
  ▼
2. Inspect Data
  │
  ▼
3. Detect Skewed Numerical Variable
  │
  ▼
4. Train Baseline Linear Regression
  │
  ▼
5. Calculate Baseline MAE, RMSE and R²
  │
  ▼
6. Apply Log Transformation
  │
  ▼
7. Compare Distribution Before and After
  │
  ▼
8. Apply StandardScaler
  │
  ▼
9. Train Second Linear Regression
  │
  ▼
10. Calculate New MAE, RMSE and R²
  │
  ▼
11. Compare Both Experiments
  │
  ▼
12. Evidence-Based Conclusion
```

------------------------------------------------------------------------

# 🤖 12. Machine Learning Model

The project uses:

``` text
LinearRegression
```

Reason:

-   Simple
-   Fast
-   Appropriate for the assignment
-   Easy to explain
-   Supports direct before/after preprocessing comparison

The project studies preprocessing rather than searching for the most
complex model.

------------------------------------------------------------------------

# 📊 13. Evaluation Metrics

Both models use the same metrics.

## MAE --- Mean Absolute Error

Average absolute difference between actual and predicted values.

**Lower is better.**

## RMSE --- Root Mean Squared Error

Prediction error that gives greater weight to larger errors.

**Lower is better.**

## R² Score

Measures how much variation in the target is explained by the model.

**Higher is generally better.**

------------------------------------------------------------------------

# 📋 14. Final Comparison

  Model                       MAE    RMSE      R²
  ----------------------- ------- ------- -------
  Before Transformation     Value   Value   Value
  After Log + Scaling       Value   Value   Value

The conclusion must be based on measured results.

Do not claim improvement unless the metrics support it.

------------------------------------------------------------------------

# 🖼️ 15. Expected Outputs

``` text
outputs/
├── before_log.png
├── after_log.png
└── model_comparison.csv
```

### Output 1

Histogram of the original skewed variable.

### Output 2

Histogram after log transformation.

### Output 3

Comparison of baseline and transformed model performance.

------------------------------------------------------------------------

# 📓 16. Notebook Structure

``` text
1. Project Title and Objective
2. Import Libraries
3. Load Dataset
4. Basic Data Inspection
5. Missing Value Analysis
6. Select Numerical Data
7. Calculate Skewness
8. Visualize Original Distribution
9. Build Baseline Regression Model
10. Evaluate Baseline Model
11. Apply Log Transformation
12. Visualize Transformed Distribution
13. Apply StandardScaler
14. Build Transformed Regression Model
15. Evaluate Transformed Model
16. Compare Results
17. Final Conclusion
```

------------------------------------------------------------------------

# 🚫 17. Scope Control

Excluded from the initial project:

-   ❌ React
-   ❌ HTML/CSS/JavaScript frontend
-   ❌ Database
-   ❌ REST API
-   ❌ Authentication
-   ❌ Docker
-   ❌ Deep Learning
-   ❌ Complex backend architecture
-   ❌ Multiple unnecessary machine-learning algorithms

## Optional Extension

If a visual application is required later, add a small Streamlit
dashboard after the core analysis is complete.

------------------------------------------------------------------------

# 🏁 18. Final Deliverables

-   📁 Project folder structure
-   📊 Dataset
-   📓 Jupyter Notebook
-   📈 Original distribution graph
-   📉 Log-transformed distribution graph
-   🤖 Baseline regression model
-   🤖 Transformed regression model
-   📋 Model comparison CSV
-   📏 MAE, RMSE and R² metrics
-   📝 Final evidence-based conclusion
-   📄 `requirements.txt`
-   📄 `README.md`

------------------------------------------------------------------------

# 🚀 Final Pipeline

``` text
DATASET
   │
   ▼
INSPECT DATA
   │
   ▼
MEASURE SKEWNESS
   │
   ├───────────────────────┐
   │                       │
   ▼                       ▼
BASELINE MODEL       LOG TRANSFORMATION
   │                       │
   ▼                       ▼
METRICS              STANDARD SCALER
   │                       │
   │                       ▼
   │                 SECOND MODEL
   │                       │
   │                       ▼
   └──────────────► COMPARE METRICS
                           │
                           ▼
                      CONCLUSION
```

## Final Principle

> **Inspect first. Transform based on evidence. Prevent data leakage.
> Compare identical evaluation metrics. Draw conclusions from measured
> results.**
