import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Titanic Survival Analysis Dashboard",
    layout="wide"
)

st.title("Titanic Survival Analysis Dashboard")
st.write("Explore survival patterns using passenger class, gender, and age.")

# Load the dataset
titanic_data = pd.read_csv("./data/titanic.csv")

# Basic data cleaning
clean_titanic_data = titanic_data.dropna().copy()

# Map numeric codes to readable labels
clean_titanic_data["SurvivalStatus"] = clean_titanic_data["Survived"].map({
    0: "Non-Survivor",
    1: "Survivor"
})

clean_titanic_data["PassengerClass"] = clean_titanic_data["Pclass"].map({
    1: "First Class",
    2: "Second Class",
    3: "Third Class"
})

# Preview the dataset
st.subheader("Dataset Preview")
st.dataframe(clean_titanic_data.head(10), width="stretch")

# Survival rates by passenger class
survival_by_class = (
    clean_titanic_data.groupby("PassengerClass")["Survived"]
    .mean()
    .mul(100)
    .reset_index(name="SurvivalRate")
)

class_order = ["First Class", "Second Class", "Third Class"]

survival_by_class["PassengerClass"] = pd.Categorical(
    survival_by_class["PassengerClass"],
    categories=class_order,
    ordered=True
)

survival_by_class = survival_by_class.sort_values("PassengerClass")
survival_by_class["SurvivalRate"] = survival_by_class["SurvivalRate"].round(2)

# Survival rates by gender
survival_by_gender = (
    clean_titanic_data.groupby("Sex")["Survived"]
    .mean()
    .mul(100)
    .reset_index(name="SurvivalRate")
)

survival_by_gender["SurvivalRate"] = survival_by_gender["SurvivalRate"].round(2)

st.subheader("Survival Rates")

first_column, second_column = st.columns(2)

with first_column:
    st.write("Survival Rate by Passenger Class")
    st.dataframe(survival_by_class, width="stretch")

with second_column:
    st.write("Survival Rate by Gender")
    st.dataframe(survival_by_gender, width="stretch")

# Visualize survival rates
st.subheader("Survival Rate by Passenger Class")

figure, axis = plt.subplots(figsize=(8, 5))

axis.bar(
    survival_by_class["PassengerClass"],
    survival_by_class["SurvivalRate"]
)

axis.set_xlabel("Passenger Class")
axis.set_ylabel("Survival Rate (%)")

st.pyplot(figure)

st.subheader("Survival Rate by Gender")

figure, axis = plt.subplots(figsize=(7, 5))

axis.bar(
    survival_by_gender["Sex"],
    survival_by_gender["SurvivalRate"]
)

axis.set_xlabel("Gender")
axis.set_ylabel("Survival Rate (%)")

st.pyplot(figure)

# Age distribution
st.subheader("Age Distribution by Survival Status")

figure, axis = plt.subplots(figsize=(8, 5))

sns.boxplot(
    data=clean_titanic_data,
    x="SurvivalStatus",
    y="Age",
    ax=axis
)

axis.set_xlabel("Survival Status")
axis.set_ylabel("Age")

st.pyplot(figure)

# Average age comparison
average_age_by_survival = (
    clean_titanic_data.groupby("SurvivalStatus")["Age"]
    .mean()
)

average_age_survivors = average_age_by_survival["Survivor"]
average_age_non_survivors = average_age_by_survival["Non-Survivor"]

st.subheader("Average Age Comparison")

first_column, second_column = st.columns(2)

first_column.metric(
    "Average Age of Survivors",
    f"{average_age_survivors:.2f} years"
)

second_column.metric(
    "Average Age of Non-Survivors",
    f"{average_age_non_survivors:.2f} years"
)

