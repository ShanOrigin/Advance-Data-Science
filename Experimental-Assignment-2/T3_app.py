import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Movie Ratings Explorer",
    layout="wide"
)

st.title("Movie Ratings Explorer")
st.write("Explore and analyse movie ratings based on different genres.")

# Load the dataset
movies_data = pd.read_csv("./data/movies.csv")

# Remove rows with missing values in key columns
clean_movies_data = movies_data.dropna(
    subset=["Genre", "Year", "Rating", "Votes"]
).copy()

# Preview the dataset
st.subheader("Dataset Preview")
st.dataframe(clean_movies_data.head(10), width="stretch")

# Select a genre
available_genres = sorted(
    clean_movies_data["Genre"].unique().tolist()
)

selected_genre = st.selectbox(
    "Select a Genre",
    available_genres
)

# Filter data for the selected genre
selected_genre_data = clean_movies_data[
    clean_movies_data["Genre"] == selected_genre
]

# Calculate summary statistics
average_rating = selected_genre_data["Rating"].mean()
average_votes = selected_genre_data["Votes"].mean()
median_release_year = selected_genre_data["Year"].median()

st.subheader(f"Summary Statistics - {selected_genre}")

first_column, second_column, third_column = st.columns(3)

first_column.metric("Average Rating", f"{average_rating:.2f}")
second_column.metric("Average Votes", f"{average_votes:.0f}")
third_column.metric(
    "Median Release Year",
    f"{median_release_year:.0f}"
)

# Boxplot for all genres
st.subheader("Movie Rating Distribution Across Genres")

figure, axis = plt.subplots(figsize=(8, 5))

sns.boxplot(
    data=clean_movies_data,
    x="Genre",
    y="Rating",
    ax=axis
)

axis.set_title("Movie Rating Distribution Across Genres")
axis.set_xlabel("Genre")
axis.set_ylabel("Rating")

st.pyplot(figure)

# Scatterplot for the selected genre
st.subheader(f"Votes vs Rating - {selected_genre}")

figure, axis = plt.subplots(figsize=(9, 5))

scatter = axis.scatter(
    selected_genre_data["Votes"],
    selected_genre_data["Rating"],
    s=selected_genre_data["Votes"] / 25,
    c=selected_genre_data["Rating"],
    alpha=0.7
)

figure.colorbar(scatter, ax=axis, label="Rating")

axis.set_title(f"Votes vs Rating - {selected_genre}")
axis.set_xlabel("Votes")
axis.set_ylabel("Rating")

st.pyplot(figure)

