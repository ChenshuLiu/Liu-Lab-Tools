import cv2
import numpy as np
import pandas as pd
import streamlit as st
from datetime import timedelta

st.title("Free Style Libre CGM Data Extraction Tool")

# Upload chart image
uploaded_file = st.file_uploader("Upload chart image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR")

    # Step 1: Calibration
    top_val = st.number_input("Enter value for top calibration line (mmol/L):", value=9.0)
    bottom_val = st.number_input("Enter value for bottom calibration line (mmol/L):", value=3.0)

    start_time = st.text_input("Enter start time (e.g. 12pm Tue):", "12pm Tue")
    end_time = st.text_input("Enter end time (e.g. 11am Wed):", "11am Wed")
    interval_min = st.number_input("Interval (minutes):", value=5)

    if st.button("Extract Data"):
        # --- Step 2: Detect curve (black line) ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # --- Step 2a: Mask for red lines ---
        # Red appears in two ranges in HSV (around 0° and 180°)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        # --- Step 2b: Mask for black lines ---
        # Black means very low brightness (V) and saturation
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)

        # --- Step 3: Combine both masks ---
        mask = cv2.bitwise_or(mask_red, mask_black)

        # Optional: clean up the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # --- Step 4: Get coordinates of all line pixels ---
        coords = np.column_stack(np.where(mask > 0))  # y, x coordinates

        # --- Step 3: Map pixels to values using full image height ---
        y_min, y_max = 0, img.shape[0]-1  # top and bottom of the image
        x_min, x_max = coords[:,1].min(), coords[:,1].max()

        def pixel_to_value(y):
            # Invert y-axis: top of image = top_val, bottom = bottom_val
            return bottom_val + (top_val - bottom_val) * (y_max - y) / (y_max - y_min)

        # Time mapping
        total_minutes = (24*60)  # example 24h, should parse actual start/end
        def pixel_to_time(x):
            return (x - x_min) / (x_max - x_min) * total_minutes

        # --- Step 4: Interpolation every interval ---
        times = np.arange(0, total_minutes+1, interval_min)
        values = []
        for t in times:
            x_target = x_min + (t/total_minutes) * (x_max - x_min)
            # nearest curve point
            nearest = coords[np.argmin(abs(coords[:,1]-x_target))]
            values.append(pixel_to_value(nearest[0]))

        df = pd.DataFrame({"minutes": times, "value": values})
        st.dataframe(df)
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", data=csv, file_name="extracted.csv")
