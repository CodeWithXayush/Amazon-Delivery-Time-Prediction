# feature_engineering.py
"""
Script: Feature Engineering for Amazon Delivery Time Prediction
Author: Ayush Kumar
Project: Amazon Delivery Time Prediction
"""

import pandas as pd
import numpy as np
import os

# -------------------------------
# Paths
# -------------------------------
DATA_FOLDER = "data"
RAW_DATA_PATH = os.path.join(DATA_FOLDER, "amazon_delivery.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_FOLDER, "amazon_delivery_processed.csv")

# -------------------------------
# Load dataset
# -------------------------------
if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at {RAW_DATA_PATH}. Please check the file path.")

df = pd.read_csv(RAW_DATA_PATH)
print("✅ Original dataset loaded successfully!")
print(f"📊 Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(df.head(3))

# -------------------------------
# Drop unnecessary columns
# -------------------------------
if "Order_ID" in df.columns:
    df.drop(columns=["Order_ID"], inplace=True)

# Drop any redundant time dummy columns
high_card_cols = [col for col in df.columns if col.startswith("Order_Time_")]
df.drop(columns=high_card_cols, errors="ignore", inplace=True)

# -------------------------------
# Parse and combine date-time
# -------------------------------
if "Order_Date" in df.columns:
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

if "Order_Time" in df.columns:
    df["Order_Time"] = pd.to_datetime(df["Order_Time"], format="%H:%M:%S", errors="coerce").dt.time

if "Order_Date" in df.columns and "Order_Time" in df.columns:
    df["Order_Timestamp"] = pd.to_datetime(df["Order_Date"].astype(str) + " " + df["Order_Time"].astype(str), errors="coerce")
    df.drop(columns=["Order_Date", "Order_Time"], inplace=True)
else:
    df.rename(columns={"Order_Date": "Order_Timestamp"}, inplace=True)

# -------------------------------
# Extract time-based features
# -------------------------------
if "Order_Timestamp" in df.columns:
    df["Hour"] = df["Order_Timestamp"].dt.hour
    df["Weekday"] = df["Order_Timestamp"].dt.weekday
    df["Month"] = df["Order_Timestamp"].dt.month
    df.drop(columns=["Order_Timestamp"], inplace=True, errors="ignore")

# -------------------------------
# Haversine Distance Calculation
# -------------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Compute Haversine distance (km) between two coordinates."""
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

if all(col in df.columns for col in ["Store_Latitude", "Store_Longitude", "Drop_Latitude", "Drop_Longitude"]):
    df["Distance_km"] = haversine(df["Store_Latitude"], df["Store_Longitude"],
                                  df["Drop_Latitude"], df["Drop_Longitude"])
    df.drop(columns=["Store_Latitude", "Store_Longitude", "Drop_Latitude", "Drop_Longitude"], inplace=True)

# -------------------------------
# Handle Missing Values
# -------------------------------
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    if not df[col].mode().empty:
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna("Unknown", inplace=True)

# -------------------------------
# Encode Categorical Variables
# -------------------------------
df = pd.get_dummies(df, drop_first=True)

# -------------------------------
# Save Processed Data
# -------------------------------
os.makedirs(DATA_FOLDER, exist_ok=True)
df.to_csv(PROCESSED_DATA_PATH, index=False)

print("\n✅ Feature Engineering Completed Successfully!")
print(f"📁 Processed data saved at: {os.path.abspath(PROCESSED_DATA_PATH)}")
print(f"📊 Final Shape: {df.shape}")
