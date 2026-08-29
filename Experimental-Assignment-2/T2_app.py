import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Student Grades Explorer", layout="wide")

st.title("Student Grades Explorer")
st.write("Explore student performance across different subjects.")

# Load the dataset
grades_data = pd.read_csv("./data/grades.csv")

# Preview the dataset
st.subheader("Dataset Preview")
st.dataframe(grades_data.head(10), width="stretch")

# Select a subject
subjects = grades_data["Subject"].unique().tolist()

selected_subject = st.selectbox(
    "Select a Subject",
    subjects
)

# Filter data for the selected subject
selected_subject_data = grades_data[
    grades_data["Subject"] == selected_subject
]

# Calculate summary statistics
mean_final_score = selected_subject_data["Final"].mean()
median_final_score = selected_subject_data["Final"].median()
standard_deviation_final_score = selected_subject_data["Final"].std()

st.subheader(f"Summary Statistics - {selected_subject}")

first_column, second_column, third_column = st.columns(3)

first_column.metric("Mean", f"{mean_final_score:.2f}")
second_column.metric("Median", f"{median_final_score:.2f}")
third_column.metric(
    "Standard Deviation",
    f"{standard_deviation_final_score:.2f}"
)

# Boxplot for all subjects
st.subheader("Final Score Distribution Across Subjects")

figure, axis = plt.subplots(figsize=(8, 5))
sns.boxplot(
    data=grades_data,
    x="Subject",
    y="Final",
    ax=axis
)
axis.set_title("Final Score Distribution Across Subjects")
axis.set_xlabel("Subject")
axis.set_ylabel("Final Score")

st.pyplot(figure)

# Scatterplot for the selected subject
st.subheader(f"Test1 vs Final Scores - {selected_subject}")

figure, axis = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    data=selected_subject_data,
    x="Test1",
    y="Final",
    hue="Subject",
    ax=axis
)
axis.set_title(f"Test1 vs Final Scores - {selected_subject}")
axis.set_xlabel("Test1 Score")
axis.set_ylabel("Final Score")

st.pyplot(figure)