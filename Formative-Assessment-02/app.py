"""
FA2 Case Study - Student Performance Prediction
PRN: 125M1H026
Student: Shantanu Suryawanshi

Run locally:
    streamlit run app.py

The app downloads the UCI Student Performance Mathematics dataset,
trains the required models if saved model files are not available,
and stores the trained pipelines with joblib.
"""

from pathlib import Path
import io
import zipfile
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_STATE = 42

UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/320/student.zip"
FALLBACK_DATASET_URL = (
    "https://raw.githubusercontent.com/gramener/datasets/main/student-mat.csv"
)

SELECTED_FEATURES = [
    "age", "Medu", "Fedu", "studytime", "failures",
    "absences", "G1", "G2", "higher", "internet"
]

NUMERIC_FEATURES = [
    "age", "Medu", "Fedu", "studytime", "failures",
    "absences", "G1", "G2"
]

CATEGORICAL_FEATURES = ["higher", "internet"]

MODEL_FOLDER = Path("models")

MODEL_FILES = {
    "Linear Regression": "linear_regression.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Tuned Random Forest": "tuned_random_forest.joblib",
}

METRICS_FILE = MODEL_FOLDER / "metrics.joblib"


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

@st.cache_data
def load_student_data():
    try:
        with urlopen(UCI_DATASET_URL, timeout=30) as response:
            zip_data = io.BytesIO(response.read())

        with zipfile.ZipFile(zip_data) as archive:
            student_file = next(
                name
                for name in archive.namelist()
                if name.endswith("student-mat.csv")
            )

            with archive.open(student_file) as file:
                data = pd.read_csv(file, sep=";")

    except Exception:
        with urlopen(FALLBACK_DATASET_URL, timeout=30) as response:
            data = pd.read_csv(response, sep=";")

    return data


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

def create_preprocessor():
    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder)
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])


# ---------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------

def train_models(data):
    data = data.copy()
    data["pass_fail"] = np.where(data["G3"] >= 10, 1, 0)

    features = data[SELECTED_FEATURES]
    regression_target = data["G3"]
    classification_target = data["pass_fail"]

    (
        regression_train,
        regression_test,
        regression_y_train,
        regression_y_test
    ) = train_test_split(
        features,
        regression_target,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    (
        classification_train,
        classification_test,
        classification_y_train,
        classification_y_test
    ) = train_test_split(
        features,
        classification_target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=classification_target
    )

    preprocessor = create_preprocessor()

    models = {}

    # Regression model
    models["Linear Regression"] = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])

    # Classification models
    models["Logistic Regression"] = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ])

    models["Decision Tree"] = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(
            max_depth=5,
            random_state=RANDOM_STATE
        ))
    ])

    models["Random Forest"] = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=150,
            random_state=RANDOM_STATE
        ))
    ])

    models["Support Vector Machine"] = Pipeline([
        ("preprocessor", preprocessor),
        ("model", SVC(
            probability=True,
            random_state=RANDOM_STATE
        ))
    ])

    models["K-Nearest Neighbors"] = Pipeline([
        ("preprocessor", preprocessor),
        ("model", KNeighborsClassifier(
            n_neighbors=7
        ))
    ])

    models["Naive Bayes"] = Pipeline([
        ("preprocessor", preprocessor),
        ("model", GaussianNB())
    ])

    # Train regression model.
    models["Linear Regression"].fit(
        regression_train,
        regression_y_train
    )

    # Train classification models.
    for model_name in [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine",
        "K-Nearest Neighbors",
        "Naive Bayes"
    ]:
        models[model_name].fit(
            classification_train,
            classification_y_train
        )

    # GridSearchCV for Random Forest.
    tuning_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            random_state=RANDOM_STATE
        ))
    ])

    parameter_grid = {
        "model__n_estimators": [100, 150],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5]
    }

    grid_search = GridSearchCV(
        tuning_pipeline,
        parameter_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(
        classification_train,
        classification_y_train
    )

    models["Tuned Random Forest"] = grid_search.best_estimator_

    # Save all trained pipelines.
    MODEL_FOLDER.mkdir(exist_ok=True)

    for model_name, model in models.items():
        joblib.dump(
            model,
            MODEL_FOLDER / MODEL_FILES[model_name]
        )

    # Save compact test-set metrics for display.
    metrics = {
        "regression": {},
        "classification": {},
        "tuning": {
            "best_parameters": grid_search.best_params_,
            "best_cv_accuracy": float(grid_search.best_score_)
        }
    }

    regression_predictions = models["Linear Regression"].predict(
        regression_test
    )

    metrics["regression"] = {
        "mae": float(
            np.mean(np.abs(regression_y_test - regression_predictions))
        )
    }

    for model_name in [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "Tuned Random Forest"
    ]:
        predictions = models[model_name].predict(classification_test)

        metrics["classification"][model_name] = {
            "accuracy": float(
                accuracy_score(classification_y_test, predictions)
            ),
            "precision": float(
                precision_score(
                    classification_y_test,
                    predictions,
                    zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    classification_y_test,
                    predictions,
                    zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    classification_y_test,
                    predictions,
                    zero_division=0
                )
            )
        }

    joblib.dump(metrics, METRICS_FILE)

    return models, metrics


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

@st.cache_resource
def load_models_and_metrics():
    data = load_student_data()

    all_model_files_exist = all(
        (MODEL_FOLDER / filename).exists()
        for filename in MODEL_FILES.values()
    )

    if all_model_files_exist and METRICS_FILE.exists():
        models = {
            model_name: joblib.load(
                MODEL_FOLDER / filename
            )
            for model_name, filename in MODEL_FILES.items()
        }

        metrics = joblib.load(METRICS_FILE)
        return data, models, metrics

    models, metrics = train_models(data)
    return data, models, metrics


# ---------------------------------------------------------------------
# Streamlit interface
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction")
st.caption(
    "FA2 Case Study | Advanced Data Science Lab [MCA33PE21]"
)

st.markdown(
    """
This application uses machine learning models trained on the **UCI
Student Performance — Mathematics** dataset.

Choose a prediction task, enter the student's details, and generate a
prediction using the trained model.
"""
)

try:
    student_data, models, metrics = load_models_and_metrics()
except Exception as error:
    st.error(
        "The application could not load the dataset or train the models."
    )
    st.code(str(error))
    st.stop()

st.sidebar.header("Prediction Settings")

prediction_task = st.sidebar.selectbox(
    "Select prediction task",
    [
        "Final Grade Prediction",
        "Pass / Fail Prediction"
    ]
)

if prediction_task == "Pass / Fail Prediction":
    selected_model_name = st.sidebar.selectbox(
        "Select classification model",
        [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Support Vector Machine",
            "K-Nearest Neighbors",
            "Naive Bayes",
            "Tuned Random Forest"
        ]
    )
else:
    selected_model_name = "Linear Regression"

st.sidebar.markdown("---")
st.sidebar.info(
    "Pass is defined as G3 ≥ 10. "
    "The final grade is measured on a 0–20 scale."
)

# ---------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------

st.subheader("Student Information")

first_column, second_column = st.columns(2)

with first_column:
    age = st.number_input(
        "Age",
        min_value=15,
        max_value=22,
        value=17
    )

    mother_education = st.slider(
        "Mother's education level (Medu)",
        min_value=0,
        max_value=4,
        value=2
    )

    father_education = st.slider(
        "Father's education level (Fedu)",
        min_value=0,
        max_value=4,
        value=2
    )

    study_time = st.slider(
        "Weekly study time",
        min_value=1,
        max_value=4,
        value=2
    )

    failures = st.slider(
        "Past class failures",
        min_value=0,
        max_value=3,
        value=0
    )

with second_column:
    absences = st.number_input(
        "School absences",
        min_value=0,
        max_value=93,
        value=5
    )

    first_period_grade = st.slider(
        "First-period grade (G1)",
        min_value=0,
        max_value=20,
        value=10
    )

    second_period_grade = st.slider(
        "Second-period grade (G2)",
        min_value=0,
        max_value=20,
        value=10
    )

    wants_higher_education = st.selectbox(
        "Wants higher education",
        ["yes", "no"]
    )

    has_internet = st.selectbox(
        "Internet access at home",
        ["yes", "no"]
    )

input_data = pd.DataFrame([{
    "age": age,
    "Medu": mother_education,
    "Fedu": father_education,
    "studytime": study_time,
    "failures": failures,
    "absences": absences,
    "G1": first_period_grade,
    "G2": second_period_grade,
    "higher": wants_higher_education,
    "internet": has_internet
}])

st.subheader("Prediction")

if st.button("Predict", type="primary"):
    selected_model = models[selected_model_name]

    if prediction_task == "Final Grade Prediction":
        predicted_grade = float(
            selected_model.predict(input_data)[0]
        )

        predicted_grade = max(
            0.0,
            min(20.0, predicted_grade)
        )

        st.metric(
            "Predicted Final Grade",
            f"{predicted_grade:.2f} / 20"
        )

        if predicted_grade >= 10:
            st.success("Prediction: Pass")
        else:
            st.warning("Prediction: Fail")

    else:
        predicted_class = int(
            selected_model.predict(input_data)[0]
        )

        if predicted_class == 1:
            st.success("Prediction: PASS")
        else:
            st.warning("Prediction: FAIL")

        if hasattr(selected_model, "predict_proba"):
            probability = selected_model.predict_proba(input_data)[0][1]
            st.metric(
                "Estimated Pass Probability",
                f"{probability * 100:.1f}%"
            )

# ---------------------------------------------------------------------
# Model information
# ---------------------------------------------------------------------

st.markdown("---")
st.subheader("Model Performance")

if prediction_task == "Final Grade Prediction":
    st.write(
        "Linear Regression is evaluated using MAE on the held-out test set."
    )

    st.metric(
        "Test MAE",
        f"{metrics['regression']['mae']:.3f}"
    )

else:
    model_metrics = metrics["classification"][selected_model_name]

    metric_one, metric_two, metric_three, metric_four = st.columns(4)

    metric_one.metric(
        "Accuracy",
        f"{model_metrics['accuracy']:.3f}"
    )
    metric_two.metric(
        "Precision",
        f"{model_metrics['precision']:.3f}"
    )
    metric_three.metric(
        "Recall",
        f"{model_metrics['recall']:.3f}"
    )
    metric_four.metric(
        "F1-Score",
        f"{model_metrics['f1']:.3f}"
    )

st.markdown("---")
st.caption(
    "Dataset: Cortez, P. (2008), Student Performance, "
    "UCI Machine Learning Repository. "
    "The application uses the Mathematics course data."
)
