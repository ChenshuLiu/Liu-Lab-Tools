import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="CA Normalization", layout="wide")
st.title("📊 CA Normalization")

uploaded_file = st.file_uploader("📤 Upload Data File", type=["csv", "xlsx"])
if uploaded_file:
    # --------------------- FILE READING ---------------------
    file_ext = uploaded_file.name.split(".")[-1].lower()
    try:
        if file_ext == "csv":
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin1")
        elif file_ext in ["xls", "xlsx"]:
            df = pd.read_excel(uploaded_file)

            # --------------------- SANITIZE NUMERIC-LOOKING COLUMNS ---------------------
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace(r"[^\d\.\-eE+]", "", regex=True)
                        .str.strip()
                    )
                    df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()

    st.write("### 📂 Raw Data Preview:")
    st.dataframe(df.head(), use_container_width=True)
    # print(df.dtypes)

    # --------------------- COLUMN SELECTION ---------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("The dataset must contain at least two numeric columns for comparison.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        col_a = st.selectbox("Select First CV Column", numeric_cols, index=0)
    with col2:
        col_b = st.selectbox("Select Second CV Column", numeric_cols, index=1)

    x_axis = st.selectbox("Select X-Axis Column (optional)", df.columns.tolist(), index=0)

    # --------------------- DATA PREPARATION ---------------------
    # Original arrays
    y1 = df[col_a].values
    y2 = df[col_b].values
    x = df[x_axis].values if np.issubdtype(df[x_axis].dtype, np.number) else np.arange(len(df))

    # Remove NaNs for safe plotting/calculation
    y1_clean = y1[~np.isnan(y1)]
    y2_clean = y2[~np.isnan(y2)]

    len1 = len(y1_clean)
    len2 = len(y2_clean)
    max_len = max(len1, len2)

    # --------------------- STANDARDIZATION ---------------------
    print(f"np.abs max is {np.max(np.abs(y1_clean))}")
    print(f"np.min is {np.min(y1_clean)}")
    max1 = np.max(np.abs(y1_clean)) if np.max(np.abs(y1_clean)) != 0 else 1
    print(f"max1 is {max1}")
    max2 = np.max(np.abs(y2_clean)) if np.max(np.abs(y2_clean)) != 0 else 1
    y1_std = -np.abs(y1_clean) / max1
    print(f"y1_std min {np.min(y1_std)}")
    y2_std = -np.abs(y2_clean) / max2
    print(f"y2_std min {np.min(y2_std)}")

    # --------------------- PLOTTING ---------------------
    st.subheader("📈 Raw CV Curves")
    fig_raw, ax_raw = plt.subplots(figsize=(10, 5))
    ax_raw.plot(x[:len1], y1_clean, label=col_a, linewidth=2)
    ax_raw.plot(x[:len2], y2_clean, label=col_b, linewidth=2)
    ax_raw.set_xlabel(x_axis)
    ax_raw.set_ylabel("Current (μA)")
    ax_raw.set_title("Raw CV Curves")
    ax_raw.legend()
    st.pyplot(fig_raw, use_container_width=True)

    st.subheader("⚙️ Standardized CV Curves")
    fig_std, ax_std = plt.subplots(figsize=(10, 5))
    ax_std.plot(x[:len1], y1_std, label=f"{col_a} (Standardized)", linewidth=2)
    ax_std.plot(x[:len2], y2_std, label=f"{col_b} (Standardized)", linewidth=2)
    ax_std.set_xlabel(x_axis)
    ax_std.set_ylabel("Standardized Current")
    ax_std.set_title("Standardized CV Curves")
    ax_std.legend()
    st.pyplot(fig_std, use_container_width=True)

    # --------------------- POINT-WISE DIFFERENCE COMPUTATION ---------------------
    # Align lengths
    min_len = min(len1, len2)
    y1_aligned = y1_std[:min_len]
    y2_aligned = y2_std[:min_len]

    print(f"y1_aligned {np.max(y1_std)} {np.max(y2_std)}")
    # Absolute point-wise differences
    abs_diffs = np.abs(y1_aligned - y2_aligned)

    # Largest absolute difference
    max_abs_diff = np.max(abs_diffs)

    st.subheader("📏 Maximum Point-wise Difference")
    st.metric(
        label="Max |y₁ − y₂| (standardized, point-wise)",
        value=f"{max_abs_diff:.4f}"
        )


    # --------------------- EXPORT ---------------------
    y1_export = np.pad(y1_std, (0, max_len - len1), constant_values=np.min(y1_std))
    y2_export = np.pad(y2_std, (0, max_len - len2), constant_values=np.min(y2_std))
    x_full = np.arange(max_len)

    df_out = pd.DataFrame({
        x_axis: x_full,
        col_a: y1_export,
        col_b: y2_export,
        f"{col_a}_standardized": y1_export,
        f"{col_b}_standardized": y2_export
    })

    st.subheader("📤 Export Standardized Data")
    csv_data = df_out.to_csv(index=False).encode('utf-8')
    st.download_button(
        "💾 Download Standardized CSV",
        data=csv_data,
        file_name="CV_standardized_output.csv",
        mime="text/csv"
    )

else:
    st.info("👈 Upload your CV CSV or XLSX file to begin.")
