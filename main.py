import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Heart Disease Data Analysis")

# Default dataset (GitHub raw link)
DEFAULT_DATASET = "https://raw.githubusercontent.com/Tanish3967/Project---Heart-Disease-Analysis/main/heart.csv"

def load_data(file=None):
    try:
        if file is not None:
            df = pd.read_csv(file)
        else:
            df = pd.read_csv(DEFAULT_DATASET)

        df.dropna(inplace=True)  # Remove missing values
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
df = load_data(uploaded_file)

st.write("### Data Preview")
st.dataframe(df.head(5))

st.write("### Data Statistics")
st.write(df.describe())

# Check for duplicate entries
st.write("### Duplicate Entries")
st.write(f"Total Duplicates: {df.duplicated().sum()}")

# Age Distribution
st.write("### Age Distribution")
fig, ax = plt.subplots()
sns.histplot(df["Age"], kde=True, color="green", edgecolor="red", ax=ax)
st.pyplot(fig)

# RestingBP Distribution
st.write("### Resting Blood Pressure Distribution")
fig, ax = plt.subplots()
sns.histplot(df["RestingBP"], kde=True, color="blue", ax=ax)
st.pyplot(fig)

# Cholesterol Distribution
st.write("### Cholesterol Levels")
fig, ax = plt.subplots()
sns.histplot(df["Cholesterol"], kde=False, color="blue", edgecolor="black", ax=ax)
st.pyplot(fig)

# Max Heart Rate Distribution
st.write("### Heart Rate Distribution")
fig, ax = plt.subplots()
sns.histplot(df["MaxHR"], kde=True, color="purple", ax=ax)
st.pyplot(fig)

# Oldpeak Distribution
st.write("### Oldpeak Distribution")
fig, ax = plt.subplots()
sns.histplot(df["Oldpeak"], kde=True, color="brown", ax=ax)
st.pyplot(fig)

# Pie Chart - Sex Distribution
st.write("### Gender Distribution")
fig, ax = plt.subplots()
df.groupby("Sex").size().plot(kind="pie", autopct="%.1f%%", legend=True, ax=ax)
st.pyplot(fig)

# Pie Chart - RestingECG Distribution
st.write("### Resting ECG Distribution")
fig, ax = plt.subplots()
df.groupby("RestingECG").size().plot(kind="pie", autopct="%.1f%%", legend=True, ax=ax)
st.pyplot(fig)

# Correlation Heatmap
st.write("### Correlation Heatmap")
numeric_df = df.select_dtypes(include=["number"])  # Select only numeric columns

if not numeric_df.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.error("No numeric columns available for correlation heatmap.")
