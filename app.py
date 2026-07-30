import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Basic page setup
st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")

st.title("🛍️ Customer Segmentation Explorer")
st.write(
    "This app uses a **K-Means clustering model** trained on mall customers data"
    "to group people into 5 spending 'segments'. Enter a customer's details on "
    "the left to see which segment they'd fall into."
)

# Loading the saved model, scaler, and data
@st.cache_resource
def load_model_and_scaler():
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return kmeans, scaler


# @st.cache_data is the same idea, but for data instead of a model object.
@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    return df


try:
    kmeans, scaler = load_model_and_scaler()
    df = load_data()
except FileNotFoundError as e:
    st.error(
        "Missing file! Make sure app.py, kmeans_model.pkl, scaler.pkl, and "
        "Mall_Customers.csv are all in the same folder.\n\n"
        f"Details: {e}"
    )
    st.stop()  # stops the app here so we don't crash further down


# Re-create the cluster labels + human-readable segment names
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
X_scaled = scaler.transform(X)
df["Cluster"] = kmeans.predict(X_scaled)

income_33, income_66 = df["Annual Income (k$)"].quantile([0.33, 0.66])
spending_33, spending_66 = df["Spending Score (1-100)"].quantile([0.33, 0.66])


def bucket(value, low_cut, high_cut):
    # Turns a number into 'Low', 'Medium', or 'High'
    if value <= low_cut:
        return "Low"
    elif value <= high_cut:
        return "Medium"
    else:
        return "High"


def label_cluster(row):
    income_level = bucket(row["Annual Income (k$)"], income_33, income_66)
    spending_level = bucket(row["Spending Score (1-100)"], spending_33, spending_66)
    return f"{income_level} Income, {spending_level} Spending"


cluster_summary = (
    df.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    .mean()
    .round(1)
)
cluster_summary["Number of Customers"] = df["Cluster"].value_counts().sort_index()
cluster_summary["Segment Name"] = cluster_summary.apply(label_cluster, axis=1)


# Sidebar: let the user enter a new customer's details
st.sidebar.header("Enter a New Customer")

income_input = st.sidebar.slider(
    "Annual Income (k$)",
    min_value=int(df["Annual Income (k$)"].min()),
    max_value=int(df["Annual Income (k$)"].max()),
    value=60,
    help="Customer's yearly income, in thousands of dollars.",
)

spending_input = st.sidebar.slider(
    "Spending Score (1-100)",
    min_value=1,
    max_value=100,
    value=50,
    help="A score the mall assigns based on how much the customer spends (1 = spends very little, 100 = spends a lot).",
)

# Turn the two numbers into a tiny 1-row DataFrame, exactly like the notebook did
new_customer = pd.DataFrame(
    {"Annual Income (k$)": [income_input], "Spending Score (1-100)": [spending_input]}
)

# Scale it using the SAME scaler the model was trained with (never re-fit here!)
new_customer_scaled = scaler.transform(new_customer)
predicted_cluster = int(kmeans.predict(new_customer_scaled)[0])
predicted_segment = cluster_summary.loc[predicted_cluster, "Segment Name"]


# Main panel: show the prediction + the chart + the summary table
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Predicted Segment")
    st.metric(label="Cluster Number", value=predicted_cluster)
    st.success(f"**{predicted_segment}**")

    st.subheader("All Segments Found")
    st.dataframe(
        cluster_summary[["Segment Name", "Number of Customers"]],
        width="stretch",
    )

with col2:
    st.subheader("Where This Customer Falls")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot every existing customer, colored by their cluster
    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue=df["Cluster"].astype(str),
        palette="tab10",
        s=60,
        ax=ax,
        legend="full",
    )

    # Highlight the new customer as a big black star so it stands out
    ax.scatter(
        income_input,
        spending_input,
        marker="*",
        s=600,
        c="black",
        edgecolors="white",
        linewidths=1.5,
        label="New Customer",
        zorder=5,
    )

    ax.set_title("Annual Income vs Spending Score (New Customer in Black)")
    ax.legend(title="Cluster")
    st.pyplot(fig)

st.divider()
st.caption()
