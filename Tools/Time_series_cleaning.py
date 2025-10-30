import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# --------------------- MAIN PAGE: PROCESSING ---------------------
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

        # --- Handle non-numeric X values ---
        x = df[x_col]
        if not np.issubdtype(x.dtype, np.number):
            try:
                x = pd.to_datetime(x)
            except Exception:
                st.warning(f"⚠️ '{x_col}' contains non-numeric values. Using row indices instead.")
                x = np.arange(len(df))

        # --- Handle non-numeric Y values ---
        y = df[y_col]
        if not np.issubdtype(y.dtype, np.number):
            st.warning(f"⚠️ '{y_col}' contains non-numeric values. Ignoring non-numeric rows.")
            y = pd.to_numeric(y, errors='coerce')
            mask = ~y.isna()
            y = y[mask].values
            x = x[mask] if len(x) == len(mask) else np.arange(len(y))
        else:
            y = y.values

        st.sidebar.subheader("🧼 Select Filters & Parameters")

        filters = {
            "Moving Average": {},
            "Median Filter": {},
            "Savitzky-Golay": {},
            "Exponential Moving Average": {},
            "Low-pass Filter": {},
            "High-pass Filter": {},
            "Band-pass Filter": {},
            "Rolling Z-Score": {}
        }

        selected_filters = st.sidebar.multiselect(
            "Select Filters (applied in chosen order)",
            options=list(filters.keys()),
            default=[]
        )

        user_order = []
        for f in selected_filters:
            user_order.append(st.sidebar.text_input(f"Order for {f}", value=str(len(user_order)+1)))

        # Gather parameters
        params = {}
        for f in selected_filters:
            if f == "Moving Average":
                params[f] = {"window": st.sidebar.number_input("MA window", 1, 100, 5)}
            elif f == "Median Filter":
                params[f] = {"window": st.sidebar.number_input("Median window (odd ≥3)", 1, 101, 5, step=2)}
            elif f == "Savitzky-Golay":
                params[f] = {
                    "window": st.sidebar.number_input("SG window (odd ≥3)", 3, 99, 11, step=2),
                    "poly": st.sidebar.number_input("SG poly order (< window)", 1, 5, 2)
                }
            elif f == "Exponential Moving Average":
                params[f] = {"alpha": st.sidebar.slider("EMA alpha (0–1)", 0.0, 1.0, 0.3)}
            elif f == "Low-pass Filter":
                params[f] = {
                    "fs": st.sidebar.number_input("LP sampling rate", 1, 10000, 1000),
                    "cutoff": st.sidebar.number_input("LP cutoff freq", 1, 500, 50),
                    "order": st.sidebar.slider("LP order", 1, 10, 4)
                }
            elif f == "High-pass Filter":
                params[f] = {
                    "fs": st.sidebar.number_input("HP sampling rate", 1, 10000, 1000),
                    "cutoff": st.sidebar.number_input("HP cutoff freq", 1, 500, 10),
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

        # Apply filters in user order
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
                processed = butter_filter(processed, [p["f_low"], p["f_high"]], p["fs"], p["order"], 'band')
            elif f == "Rolling Z-Score":
                processed = rolling_zscore(processed, p["window"], p["threshold"])

        # Plot
        st.subheader("📊 Visualization")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x, y, label="Raw Signal", alpha=0.7)
        if selected_filters:
            ax.plot(x, processed, label="Processed Signal", linewidth=2)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        st.pyplot(fig, use_container_width=True)

        # Export
        if st.button("📥 Export Processed Data"):
            df_out = df.copy()
            df_out[y_col] = processed
            csv = df_out.to_csv(index=False)
            st.download_button("Download CSV", data=csv, file_name="processed_output.csv", mime="text/csv")

    else:
        st.info("👈 Upload a CSV file to begin.")

# --------------------- PAGE: GUIDE ---------------------
else:
    st.title("📘 Filter Guide")
    st.markdown("""
    This guide helps you select the right **time-domain filters** for your signal.

    ### 🧱 Filters Overview

    | Filter Name | Description | Best For | Key Parameters | Limitations |
    |--------------|-------------|-----------|----------------|--------------|
    | **Moving Average** | Smooths local fluctuations by averaging nearby points. | Simple noise reduction when signal varies slowly. | `window`: number of samples to average. | Blurs peaks and edges; not effective for impulse noise. |
    | **Median Filter** | Replaces each value with the median in a moving window. | Removing spikes, impulse noise, or sensor glitches. | `window`: odd integer window length. | May distort narrow peaks or fine waveform details. |
    | **Savitzky–Golay Filter** | Fits a polynomial to local segments to smooth while preserving shape. | Smoothing noisy signals with meaningful waveform (e.g., ECG, vibration). | `window`: odd integer; `poly`: polynomial order (1–5). | Sensitive to outliers; window must be larger than `poly+1`. |
    | **Exponential Moving Average (EMA)** | Weighted smoothing that reacts faster to changes. | Tracking slowly varying trends in real-time data. | `alpha`: smoothing factor (0–1). | Not ideal for strong noise or spikes. |
    | **Low-pass Filter (Butterworth)** | Allows low frequencies, attenuates high-frequency noise. | Removing high-frequency noise, e.g., sensor jitter. | `cutoff`: Hz; `fs`: sampling rate; `order`: 2–6 typical. | Can cause lag near cutoff; not effective for baseline drift. |
    | **High-pass Filter (Butterworth)** | Removes low-frequency drift and baseline wander. | Correcting slow drift or motion artifacts. | `cutoff`: Hz; `fs`: sampling rate; `order`: 2–6 typical. | May distort slow physiological signals. |
    | **Band-pass Filter (Butterworth)** | Keeps frequencies within a specific range. | Isolating frequency bands (e.g., heart rate, vibration mode). | `f_low`, `f_high`: band edges; `fs`: sampling rate; `order`. | Requires known frequency range; can distort edges if poorly tuned. |
    | **Rolling Z-Score** | Detects and replaces statistical outliers based on rolling mean/std. | Removing rare spikes without distorting general shape. | `window`: length for statistics; `threshold`: z-score limit. | Can oversmooth signals with frequent spikes. |

    ---

    ### ⚙️ Recommended Recepies

    | Scenario | Recommended Filters |
    |-----------|--------------------|
    | Sudden spikes or outliers | **Median Filter** → **Moving Average** or **Rolling Z-Score** |
    | High-frequency noise | **Low-pass Filter** or **Moving Average**, optionally combine with **High-pass Filter** |
    | Baseline drift | **High-pass Filter**, optionally followed by **Low-pass** or **Savitzky–Golay** |
    | Smooth signal with preserved peaks | **Savitzky–Golay**, optionally followed by **Exponential Moving Average** |
    | Real-time tracking or adaptive smoothing | **Exponential Moving Average**, optionally preceded by **Median Filter** |
    | Known frequency range of interest | **Band-pass Filter**, optionally combined with **Savitzky–Golay** |
    
    ---

    You can mix filters and control their **order** to fine-tune your signal pipeline.
    """)
