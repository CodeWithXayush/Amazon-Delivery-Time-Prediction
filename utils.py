from __future__ import annotations
import math
from datetime import datetime
from typing import Optional, Tuple
import pandas as pd
import numpy as np

# Categorical standardization maps
WEATHER_MAP = {
    "sunny": "Sunny", "clear": "Sunny", "clear sky": "Sunny",
    "cloudy": "Cloudy", "overcast": "Cloudy", "partly cloudy": "Cloudy",
    "rain": "Rain", "rainy": "Rain", "showers": "Rain",
    "storm": "Storm", "thunderstorm": "Storm",
}
TRAFFIC_MAP = {
    "low": "Low", "light": "Low",
    "medium": "Medium", "moderate": "Medium",
    "high": "High", "heavy": "High",
}
VEHICLE_MAP = {
    "bike": "Bike", "bicycle": "Bike",
    "scooter": "Scooter",
    "motorcycle": "Motorcycle", "motorbike": "Motorcycle",
    "car": "Car", "van": "Car",
}
AREA_MAP = {
    "urban": "Urban", "metropolitan": "Metropolitan", "metro": "Metropolitan"
}

def _norm_str(x: Optional[str]) -> str:
    return str(x).strip().lower() if pd.notnull(x) else ""

def standardize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Weather" in df.columns:
        df["Weather"] = df["Weather"].map(lambda x: WEATHER_MAP.get(_norm_str(x), str(x)))
    if "Traffic" in df.columns:
        df["Traffic"] = df["Traffic"].map(lambda x: TRAFFIC_MAP.get(_norm_str(x), str(x)))
    if "Vehicle" in df.columns:
        df["Vehicle"] = df["Vehicle"].map(lambda x: VEHICLE_MAP.get(_norm_str(x), str(x)))
    if "Area" in df.columns:
        df["Area"] = df["Area"].map(lambda x: AREA_MAP.get(_norm_str(x), str(x)))
    return df

def parse_datetime(date_str: str, time_str: Optional[str]) -> Optional[datetime]:
    if pd.isna(date_str):
        return None
    # Try multiple formats
    fmts = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d"]
    t_fmts = ["%H:%M:%S", "%H:%M"]
    d = None
    for f in fmts:
        try:
            d = datetime.strptime(str(date_str), f)
            break
        except Exception:
            continue
    if d is None:
        return None
    if time_str is None or pd.isna(time_str) or str(time_str).strip() == "":
        # default to midnight if missing
        return d
    for tf in t_fmts:
        try:
            t = datetime.strptime(str(time_str), tf).time()
            return datetime.combine(d.date(), t)
        except Exception:
            continue
    return d

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0  # km
    try:
        φ1, λ1, φ2, λ2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    except Exception:
        return np.nan
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def add_time_and_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse order and pickup datetimes
    order_dt = df.apply(lambda r: parse_datetime(r.get("Order_Date"), r.get("Order_Time")), axis=1)
    pickup_dt = df.apply(lambda r: parse_datetime(r.get("Order_Date"), r.get("Pickup_Time")), axis=1)
    df["order_dt"] = order_dt
    df["pickup_dt"] = pickup_dt

    # Time to pickup in minutes
    df["time_to_pickup_min"] = (
        (df["pickup_dt"] - df["order_dt"]).dt.total_seconds() / 60.0
    )
    df["time_to_pickup_min"] = df["time_to_pickup_min"].fillna(df["time_to_pickup_min"].median())

    # Hour and weekday
    df["order_hour"] = df["order_dt"].dt.hour.fillna(0).astype(int)
    df["order_wday"] = df["order_dt"].dt.weekday.fillna(0).astype(int)

    # Distance
    df["distance_km"] = df.apply(
        lambda r: haversine_km(
            r.get("Store_Latitude"), r.get("Store_Longitude"),
            r.get("Drop_Latitude"), r.get("Drop_Longitude")
        ),
        axis=1
    )
    # Handle edge cases
    df["distance_km"] = df["distance_km"].fillna(df["distance_km"].median())

    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop duplicate Order_ID rows, keep first
    if "Order_ID" in df.columns:
        df = df.drop_duplicates(subset=["Order_ID"], keep="first")

    # Standardize categoricals
    df = standardize_categoricals(df)

    # Basic numeric coercions
    for col in ["Agent_Age", "Agent_Rating", "Store_Latitude", "Store_Longitude",
                "Drop_Latitude", "Drop_Longitude", "Delivery_Time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle missing numeric with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    # Handle missing categorical with "Unknown"
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown")

    return df
