# Amazon-Delivery-Time-Prediction

Overview
This project predicts delivery times for Amazon e-commerce orders using machine learning regression models. It leverages features such as distance, traffic, weather, product category, and agent details to estimate how long an order will take to be delivered. The workflow covers everything from data cleaning and EDA to model building, tracking (MLflow), and deployment (Streamlit).
Skills Gained
• Python scripting and automation
• Data cleaning and preprocessing
• Exploratory Data Analysis (EDA)
• Regression modeling (Linear, Random Forest, Gradient Boosting)
• Model tracking using MLflow
• Streamlit app development for user interface
Domain
E-Commerce and Logistics
Problem Statement
Amazon handles thousands of orders daily under various conditions such as traffic, weather, agent rating, and delivery area. This project aims to predict delivery time (in hours) for each order using historical data, helping to optimize logistics operations, enhance customer experience, and improve agent efficiency.
Business Use Cases
1. Enhanced Delivery Logistics: Predict delivery times to improve customer satisfaction and optimize scheduling.
2. Dynamic Adjustments: Adjust delivery estimates based on traffic and weather.
3. Agent Performance Evaluation: Identify efficient or underperforming delivery agents.
4. Operational Efficiency: Optimize routes and resource allocation.

# Project Workflow
1. Data Preparation – Load and clean data.
2. Feature Engineering – Compute distance, extract time features.
3. EDA – Visualize trends and correlations.
4. Modeling – Train regression models.
5. Model Tracking – Log results in MLflow.
6. App Development – Build Streamlit UI.
7. Deployment – Deploy Streamlit app.
   
# Dataset Description
File: amazon_delivery.csv
Contains columns such as Order_ID, Agent_Age, Agent_Rating, Store/Drop Coordinates, Order/Delivery times, Weather, Traffic, Vehicle, Area, Category, and Delivery_Time (target).
# Key Visualizations
• Bar charts: Delivery time by product category
• Box plots: Delivery time by weather/traffic
• Scatter plots: Distance vs Delivery time
• Heatmaps: Feature correlations

<img width="983" height="450" alt="image" src="https://github.com/user-attachments/assets/cd48ecd4-1003-4822-897c-f446387f05df" />

# Results
✓ Cleaned and processed dataset
✓ Regression models trained and evaluated
✓ MLflow tracking implemented
✓ Streamlit app for prediction
Evaluation Metrics
• RMSE – Root Mean Squared Error
• MAE – Mean Absolute Error
• R² – Coefficient of Determination
# Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, MLflow, Streamlit
# Deliverables
• Clean dataset
• Regression models
• Streamlit app
• MLflow tracking logs
• Documentation and notebooks

<img width="991" height="991" alt="image" src="https://github.com/user-attachments/assets/c59c5e4d-9a2b-4204-9fd5-827d36b7c236" />


# 14-Day Timeline
Day 1–2: Import dataset & understand structure
Day 3–4: Data cleaning & preprocessing
Day 5–6: EDA & visualization
Day 7–8: Feature engineering
Day 9–10: Model training (Linear, RF, GB)
Day 11: MLflow setup & tracking
Day 12–13: Streamlit app development
Day 14: Testing & deployment

# Run Instructions
1. Clone the repository:
   git clone https://github.com/<your-username>/Amazon-Delivery-Time-Prediction.git
   cd Amazon-Delivery-Time-Prediction

2. Install dependencies:
   pip install -r requirements.txt

3. Run Streamlit app:
   streamlit run src/app.py

4. Open browser:
   http://localhost:8501
# License
This project is licensed under the MIT License — free to use and modify for research or learning purposes.
