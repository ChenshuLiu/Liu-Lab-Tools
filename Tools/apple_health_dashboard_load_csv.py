import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Apple Health CSV Explorer", layout="wide")

# ---------------------------
# Utility functions
# ---------------------------
@st.cache_data
def load_csv(file):
    """Efficient CSV loading"""
    df = pd.read_csv(file, parse_dates=["day"])
    return df

def make_summary(df):
    daily_counts = df.groupby("day").size().reset_index(name="samples")

    sampling_summary = {
        "avg_samples_per_day": daily_counts["samples"].mean(),
        "std_samples_per_day": daily_counts["samples"].std(),
        "min_samples_per_day": daily_counts["samples"].min(),
        "max_samples_per_day": daily_counts["samples"].max(),
        "total_days": daily_counts["day"].nunique(),
        "total_samples": daily_counts["samples"].sum(),
    }

    value_summary = {
        "min_value": df["value"].min(),
        "max_value": df["value"].max(),
        "mean_value": df["value"].mean(),
        "std_value": df["value"].std(),
        "median_value": df["value"].median(),
    }

    return {**sampling_summary, **value_summary}

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Settings")

# Let user enter the folder path
folder = st.sidebar.text_input(
    "Enter folder path containing CSV files:",
    value="",  # leave empty by default
    placeholder="folder directory to Apple Health CSV files"
)

if folder and os.path.isdir(folder):
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    
    if csv_files:
        indicator = st.sidebar.selectbox("Select indicator", csv_files)

        # Load selected CSV
        path = os.path.join(folder, indicator)
        df = load_csv(path)

        # Date range filter
        min_date = df["day"].min()
        max_date = df["day"].max()
        start_date, end_date = st.sidebar.date_input(
            "Select date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        df_filtered = df[(df["day"] >= pd.to_datetime(start_date)) & (df["day"] <= pd.to_datetime(end_date))]

        st.sidebar.markdown(f"Loaded **{len(df_filtered)} records** for {indicator.replace('.csv','')}")

        # Plot selection
        st.sidebar.subheader("Select plots to show")
        show_cumulative = st.sidebar.checkbox("Cumulative Samples", True)
        show_distribution = st.sidebar.checkbox("Value Distribution", True)
        show_daily_box = st.sidebar.checkbox("Daily Sampling Rate", True)
        show_summary_table = st.sidebar.checkbox("Summary Statistics", True)

        # ---------------------------
        # Main App
        # ---------------------------
        st.title("📊 Apple Health CSV Explorer")

        if show_summary_table:
            st.subheader("📋 Summary Statistics")
            summary = make_summary(df_filtered)
            st.dataframe(pd.DataFrame([summary]), use_container_width=False)

        # Daily counts
        daily_counts = df_filtered.groupby("day").size().reset_index(name="samples")
        daily_counts = daily_counts.sort_values("day")
        daily_counts["cumulative"] = daily_counts["samples"].cumsum()

        if show_cumulative:
            st.subheader("📈 Cumulative Samples (Longitudinal Richness)")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(daily_counts["day"], daily_counts["cumulative"], label=indicator.replace(".csv", ""))
            ax.set_xlabel("Date")
            ax.set_ylabel("Cumulative Samples")
            ax.set_title("Cumulative Samples Over Time")
            st.pyplot(fig, use_container_width=True)

        if show_distribution:
            st.subheader("📦 Value Distribution")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(df_filtered["value"], bins=40, kde=True, ax=ax)
            ax.set_title(f"Distribution of {indicator.replace('.csv', '')}")
            st.pyplot(fig, use_container_width=True)

        if show_daily_box:
            st.subheader("📊 Daily Sampling Rate Distribution")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.boxplot(x=daily_counts["samples"], ax=ax)
            ax.set_title("Daily Sampling Rates")
            ax.set_xlabel("Samples per Day")
            st.pyplot(fig, use_container_width=True)

    else:
        st.warning("No CSV files found in this folder.")
elif folder:
    st.error("The specified path is not a valid folder.")
