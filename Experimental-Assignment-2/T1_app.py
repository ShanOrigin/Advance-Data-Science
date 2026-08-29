import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Retail Sales Dashboard", layout="wide")

st.title("Retail Sales Dashboard")
st.write("Daily retail sales analysis and product performance overview")

# Load the dataset
sales_data = pd.read_csv("./data/sales.csv")
sales_data["Date"] = pd.to_datetime(sales_data["Date"])

# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(sales_data.head(9), use_container_width=True)

# Summary statistics
mean_daily_revenue = sales_data["Revenue"].mean()
median_daily_revenue = sales_data["Revenue"].median()

st.subheader("Revenue Summary")

first_column, second_column = st.columns(2)

first_column.metric("Mean Daily Revenue", f"{mean_daily_revenue:.1f}")
second_column.metric("Median Daily Revenue", f"{median_daily_revenue:.1f}")

# Product category summary
category_summary = (
    sales_data.groupby("ProductCategory")[["UnitsSold", "Revenue"]]
    .sum()
    .reset_index()
    .sort_values("Revenue", ascending=False)
)

st.subheader("Product Category Performance")
st.dataframe(category_summary, use_container_width=True)

# Daily revenue trend
daily_revenue = sales_data.groupby("Date")["Revenue"].sum().reset_index()

st.subheader("Daily Revenue Trend")

figure, axis = plt.subplots(figsize=(9, 5))
axis.plot(daily_revenue["Date"], daily_revenue["Revenue"], marker="o")
axis.set_title("Daily Revenue Trend")
axis.set_xlabel("Date")
axis.set_ylabel("Revenue")
plt.xticks(rotation=44)
plt.tight_layout()

st.pyplot(figure)

# Revenue by category
st.subheader("Total Revenue by Product Category")

figure, axis = plt.subplots(figsize=(7, 5))
axis.bar(category_summary["ProductCategory"], category_summary["Revenue"])
axis.set_title("Revenue by Product Category")
axis.set_xlabel("Product Category")
axis.set_ylabel("Total Revenue")
plt.tight_layout()

st.pyplot(figure)