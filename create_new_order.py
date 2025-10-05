# create_new_orders.py
"""
Script: Create sample new orders for delivery time prediction
Author: Ayush Kumar
Project: Amazon Delivery Time Prediction
"""

import pandas as pd
import os
import random
from datetime import datetime

# Folder and file setup
DATA_FOLDER = "data"
NEW_ORDERS_FILE = "new_orders.csv"
NEW_ORDERS_PATH = os.path.join(DATA_FOLDER, NEW_ORDERS_FILE)

# Ensure 'data' folder exists
os.makedirs(DATA_FOLDER, exist_ok=True)

# Predefined lists for random sampling
vehicles = ["scooter", "motorcycle", "car", "bicycle"]
areas = ["Urban", "Semi-Urban", "Metropolitan"]
categories = ["Electronics", "Clothing", "Groceries", "Household", "Books"]
weathers = ["Sunny", "Rainy", "Cloudy"]
traffic_conditions = ["Low", "Medium", "High"]

# Generate random new orders
def generate_new_orders(num_orders=10):
    new_orders = []
    for _ in range(num_orders):
        store_lat = round(random.uniform(12.90, 13.00), 4)
        store_long = round(random.uniform(77.55, 77.65), 4)
        drop_lat = store_lat + round(random.uniform(-0.03, 0.03), 4)
        drop_long = store_long + round(random.uniform(-0.03, 0.03), 4)

        new_orders.append({
            "Order_ID": f"NEW_{random.randint(1000, 9999)}",
            "Agent_Age": random.randint(22, 45),
            "Agent_Rating": round(random.uniform(3.5, 5.0), 1),
            "Store_Latitude": store_lat,
            "Store_Longitude": store_long,
            "Drop_Latitude": drop_lat,
            "Drop_Longitude": drop_long,
            "Distance": round(random.uniform(2.0, 15.0), 2),
            "Weather": random.choice(weathers),
            "Traffic": random.choice(traffic_conditions),
            "Vehicle": random.choice(vehicles),
            "Area": random.choice(areas),
            "Category": random.choice(categories),
            "Order_Date": datetime.now().strftime("%Y-%m-%d"),
            "Order_Time": datetime.now().strftime("%H:%M:%S")
        })
    return pd.DataFrame(new_orders)

# Create and save the dataset
df_new_orders = generate_new_orders(num_orders=10)
df_new_orders.to_csv(NEW_ORDERS_PATH, index=False)

print("✅ New orders dataset created successfully!")
print(f"📁 File saved at: {os.path.abspath(NEW_ORDERS_PATH)}")
print(df_new_orders.head())
