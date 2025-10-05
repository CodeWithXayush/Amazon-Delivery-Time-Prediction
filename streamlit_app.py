import os
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime, time
from utils import add_time_and_distance_features, standardize_categoricals

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "best_model.joblib")
EDA_DIR = os.path.join(BASE_DIR, "outputs", "eda")

st.set_page_config(page_title="Delivery Time Prediction", page_icon="⏱", layout="centered")

st.title("Delivery Time Prediction")
st.caption("Enter order details to estimate delivery time (hours).")

# Load model if present
pipe = None
if os.path.exists(MODEL_PATH):
    pipe = joblib.load(MODEL_PATH)
    st.success(f"Model loaded: {os.path.basename(MODEL_PATH)}")
else:
    st.warning("No trained model found. Please run train_models.py to create artifacts/best_model.joblib.")

with st.expander("Batch Predictions (Upload CSV)"):
    file = st.file_uploader("Upload a CSV with the required columns", type=["csv"])
    if file is not None and pipe is not None:
        df_in = pd.read_csv(file)
        # Try to standardize/engineer if raw-like
        df_in = standardize_categoricals(df_in)
        if not {"distance_km", "time_to_pickup_min", "order_hour", "order_wday"}.issubset(df_in.columns):
            df_in = add_time_and_distance_features(df_in)
        preds = pipe.predict(df_in)
        out = df_in.copy()
        out["predicted_delivery_time"] = preds
        st.dataframe(out.head(20))
        st.download_button("Download predictions CSV", data=out.to_csv(index=False), file_name="predictions.csv", mime="text/csv")

st.subheader("Single Prediction")
col1, col2 = st.columns(2)

with col1:
    agent_age = st.number_input("Agent Age", min_value=16, max_value=80, value=30)
    agent_rating = st.number_input("Agent Rating", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
    store_lat = st.number_input("Store Latitude", value=12.9716, format="%.6f")
    store_lon = st.number_input("Store Longitude", value=77.5946, format="%.6f")
    drop_lat = st.number_input("Drop Latitude", value=12.9352, format="%.6f")
    drop_lon = st.number_input("Drop Longitude", value=77.6245, format="%.6f")

with col2:
    order_date = st.date_input("Order Date", value=datetime.now().date())
    order_time = st.time_input("Order Time", value=time(12, 0))
    pickup_time = st.time_input("Pickup Time", value=time(12, 20))
    weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rain", "Storm", "Unknown"], index=0)
    traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Unknown"], index=1)
    vehicle = st.selectbox("Vehicle", ["Bike", "Scooter", "Motorcycle", "Car", "Unknown"], index=0)
    area = st.selectbox("Area", ["Urban", "Metropolitan", "Unknown"], index=0)
    category = st.text_input("Category", value="General")

if st.button("Predict Delivery Time"):
    single = pd.DataFrame([{
        "Agent_Age": agent_age,
        "Agent_Rating": agent_rating,
        "Store_Latitude": store_lat,
        "Store_Longitude": store_lon,
        "Drop_Latitude": drop_lat,
        "Drop_Longitude": drop_lon,
        "Order_Date": order_date.strftime("%Y-%m-%d"),
        "Order_Time": order_time.strftime("%H:%M:%S"),
        "Pickup_Time": pickup_time.strftime("%H:%M:%S"),
        "Weather": weather,
        "Traffic": traffic,
        "Vehicle": vehicle,
        "Area": area,
        "Category": category,
    }])

    single = standardize_categoricals(single)
    single = add_time_and_distance_features(single)

    if pipe is None:
        st.error("Model not loaded. Train a model first.")
    else:
        pred = float(pipe.predict(single)[0])
        st.success(f"Estimated Delivery Time: {pred:.2f} hours")

st.divider()
st.subheader("EDA Snapshots")
if os.path.exists(EDA_DIR):
    imgs = sorted([f for f in os.listdir(EDA_DIR) if f.lower().endswith(".png")])
    if imgs:
        for f in imgs:
            st.image(os.path.join(EDA_DIR, f), caption=f, use_container_width=True)
    else:
        st.info("No EDA images found. Run eda.py to generate plots.")
else:
    st.info("EDA output directory not found. Run eda.py to generate plots.")
