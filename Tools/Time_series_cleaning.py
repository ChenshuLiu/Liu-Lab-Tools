import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt, savgol_filter, medfilt

st.set_page_config(page_title="Signal Processing Tool", layout="wide")

# --------------------- FILTER FUNCTIONS ---------------------

def moving_average(signal, window):
    if window < 1:
        st.warning("⚠️ Moving Average: window must be ≥ 1. Using 3.")
        window = 3
    return np.convolve(signal, np.ones(int(window)) / int(window), mode='same')

def median_filter(signal, window):
    if window < 3 or window % 2 == 0:
        st.warning("⚠️ Median Filter: window must be odd and ≥ 3. Using 3.")
        window = 3
    return medfilt(signal, kernel_size=int(window))

def savgolay_filter(signal, window, poly):
    if window < 3 or window % 2 == 0 or window >= len(signal):
        st.warning("⚠️ Savitzky–Golay: window must be odd, ≥3, and < signal length. Using 5.")
        window = 5
    if poly >= window:
        st.warning("⚠️ Savitzky–Golay: polynomial order must be < window length. Using 2.")
        poly = 2
    return savgol_filter(signal, window_length=int(window), polyorder=int(poly))

def ema_filter(signal, alpha):
    if not (0 < alpha <= 1):
        st.warning("⚠️ EMA: alpha must be between 0 and 1. Using 0.3.")
        alpha = 0.3
    filtered = np.zeros_like(signal)
    filtered[0] = signal[0]
    for i in range(1, len(signal)):
        filtered[i] = alpha * signal[i] + (1 - alpha) * filtered[i - 1]
    return filtered

def butter_filter(signal, cutoff, fs, order=4, btype='low'):
    nyq = 0.5 * fs
    if isinstance(cutoff, (list, tuple)):
        if cutoff[0] >= cutoff[1]:
            st.warning("⚠️ Band-pass: low cutoff must be < high cutoff. Swapping values.")
            cutoff = sorted(cutoff)
        if cutoff[1] >= nyq:
            st.warning("⚠️ Cutoff above Nyquist. Reducing to Nyquist/2.")
            cutoff = [cutoff[0], nyq / 2]
        normal_cutoff = np.array(cutoff) / nyq
    else:
        if cutoff >= nyq:
            st.warning("⚠️ Cutoff above Nyquist. Reducing to Nyquist/2.")
            cutoff = nyq / 2
        normal_cutoff = cutoff / nyq

    try:
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)
        return filtfilt(b, a, signal)
    except ValueError:
        st.warning(f"⚠️ Invalid filter parameters for {btype}-pass. Skipping filter.")
        return signal

def rolling_zscore(signal, window, threshold):
    if window < 3:
        st.warning("⚠️ Rolling Z-Score: window must be ≥3. Using 10.")
        window = 10
    if threshold <= 0:
        st.warning("⚠️ Rolling Z-Score: threshold must be >0. Using 3.0.")
        threshold = 3.0

    s = pd.Series(signal)
    roll_mean = s.rolling(int(window)).mean()
    roll_std = s.rolling(int(window)).std()
    z = (s - roll_mean) / roll_std
    mask = z.abs() > threshold
    s[mask] = roll_mean[mask]
    return s.fillna(method='bfill').fillna(method='ffill').values

# --------------------- PAGE SETUP ---------------------

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["📈 Signal Processing", "📘 Filter Guide"])

# --------------------- MAIN PAGE ---------------------
if page == "📈 Signal Processing":
    st.title("📈 Signal Processing & Visualization Tool")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("### 📂 Raw Data Preview:")
        st.dataframe(df.head(), use_container_width=True)

        all_columns = df.columns.tolist()
        x_col = st.selectbox("Select X-Axis Column", all_columns)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("No numeric columns available for Y-axis.")
            st.stop()

        y_col = st.selectbox("Select Y-Axis Column", numeric_cols)

        # Handle X values
        x = df[x_col]
        if not np.issubdtype(x.dtype, np.number):
            try:
                x = pd.to_datetime(x)
            except Exception:
                st.warning(f"⚠️ '{x_col}' is non-numeric. Using row index.")
                x = np.arange(len(df))

        # Handle Y values
        y = df[y_col]
        if not np.issubdtype(y.dtype, np.number):
            st.warning(f"⚠️ '{y_col}' contains non-numeric values.")
            y = pd.to_numeric(y, errors='coerce')
            mask = ~y.isna()
            y = y[mask].values
            x = x[mask] if len(x) == len(mask) else np.arange(len(y))
        else:
            y = y.values

        # ---------------- Sidebar Controls ----------------
        st.sidebar.subheader("🧼 Filters & Parameters")

        filters = [
            "Moving Average",
            "Median Filter",
            "Savitzky-Golay",
            "Exponential Moving Average",
            "Low-pass Filter",
            "High-pass Filter",
            "Band-pass Filter",
            "Rolling Z-Score"
        ]

        selected_filters = st.sidebar.multiselect(
            "Select Filters",
            options=filters,
            default=[]
        )

        show_raw = st.sidebar.checkbox("Show Raw Signal", True)
        show_processed = st.sidebar.checkbox("Show Processed Signal", True)

        user_order = []
        for f in selected_filters:
            user_order.append(
                st.sidebar.text_input(f"Order for {f}", value=str(len(user_order) + 1))
            )

        params = {}
        for f in selected_filters:
            if f == "Moving Average":
                params[f] = {"window": st.sidebar.number_input("MA window", 1, 100, 5)}

            elif f == "Median Filter":
                params[f] = {"window": st.sidebar.number_input("Median window", 1, 101, 5, step=2)}

            elif f == "Savitzky-Golay":
                params[f] = {
                    "window": st.sidebar.number_input("SG window", 3, 99, 11, step=2),
                    "poly": st.sidebar.number_input("SG poly", 1, 5, 2)
                }

            elif f == "Exponential Moving Average":
                params[f] = {"alpha": st.sidebar.slider("EMA alpha", 0.0, 1.0, 0.3)}

            elif f == "Low-pass Filter":
                params[f] = {
                    "fs": st.sidebar.number_input("LP sampling rate", 1, 10000, 1000),
                    "cutoff": st.sidebar.number_input("LP cutoff", 1, 500, 50),
                    "order": st.sidebar.slider("LP order", 1, 10, 4)
                }

            elif f == "High-pass Filter":
                params[f] = {
                    "fs": st.sidebar.number_input("HP sampling rate", 1, 10000, 1000),
                    "cutoff": st.sidebar.number_input("HP cutoff", 1, 500, 10),
                    "order": st.sidebar.slider("HP order", 1, 10, 4)
                }

            elif f == "Band-pass Filter":
                params[f] = {
                    "fs": st.sidebar.number_input("BP sampling rate", 1, 10000, 1000),
                    "f_low": st.sidebar.number_input("BP low freq", 1, 500, 5),
                    "f_high": st.sidebar.number_input("BP high freq", 1, 500, 50),
                    "order": st.sidebar.slider("BP order", 1, 10, 4)
                }

            elif f == "Rolling Z-Score":
                params[f] = {
                    "window": st.sidebar.number_input("Z window", 5, 500, 50),
                    "threshold": st.sidebar.slider("Z threshold", 0.5, 5.0, 3.0)
                }

        # ---------------- Apply Filters ----------------
        order_dict = {f: int(user_order[i]) for i, f in enumerate(selected_filters)}
        ordered_filters = sorted(order_dict.items(), key=lambda x: x[1])

        processed = y.copy()

        for f, _ in ordered_filters:
            p = params[f]
            if f == "Moving Average":
                processed = moving_average(processed, p["window"])
            elif f == "Median Filter":
                processed = median_filter(processed, p["window"])
            elif f == "Savitzky-Golay":
                processed = savgolay_filter(processed, p["window"], p["poly"])
            elif f == "Exponential Moving Average":
                processed = ema_filter(processed, p["alpha"])
            elif f == "Low-pass Filter":
                processed = butter_filter(processed, p["cutoff"], p["fs"], p["order"], 'low')
            elif f == "High-pass Filter":
                processed = butter_filter(processed, p["cutoff"], p["fs"], p["order"], 'high')
            elif f == "Band-pass Filter":
                processed = butter_filter(
                    processed,
                    [p["f_low"], p["f_high"]],
                    p["fs"],
                    p["order"],
                    'band'
                )
            elif f == "Rolling Z-Score":
                processed = rolling_zscore(processed, p["window"], p["threshold"])

        # ---------------- Interactive Plot ----------------
        st.subheader("📊 Interactive Visualization")

        fig = go.Figure()

        if show_raw:
            fig.add_trace(go.Scattergl(
                x=x,
                y=y,
                mode="lines",
                name="Raw Signal",
                opacity=0.6,
                hovertemplate="X: %{x}<br>Y: %{y:.3f}<extra></extra>"
            ))

        if selected_filters and show_processed:
            fig.add_trace(go.Scattergl(
                x=x,
                y=processed,
                mode="lines",
                name="Processed Signal",
                line=dict(width=3),
                hovertemplate="X: %{x}<br>Y: %{y:.3f}<extra></extra>"
            ))

        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            hovermode="x unified",
            height=600,
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------- Export ----------------
        if st.button("📥 Export Processed Data"):
            df_out = df.copy()
            df_out[y_col] = processed
            csv = df_out.to_csv(index=False)
            st.download_button(
                "Download CSV",
                data=csv,
                file_name="processed_output.csv",
                mime="text/csv"
            )

    else:
        st.info("👈 Upload a CSV file to begin.")

# --------------------- GUIDE PAGE ---------------------
else:
    st.title("📘 Filter Guide")
    st.markdown("""
    This guide helps you select the right filters for your signal.

    Filters available:
    - Moving Average
    - Median Filter
    - Savitzky–Golay
    - EMA
    - Low-pass
    - High-pass
    - Band-pass
    - Rolling Z-score

    Combine filters and tune parameters for best results.
    """)