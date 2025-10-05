import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Optional: MLflow and XGBoost
try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def metrics_dict(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}

def main():
    base_dir = os.path.dirname(__file__)
    default_feat = os.path.join(base_dir, "outputs", "features.csv")
    default_proc = os.path.join(base_dir, "outputs", "processed.csv")
    in_path = sys.argv[1] if len(sys.argv) > 1 else (default_feat if os.path.exists(default_feat) else default_proc)
    art_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(art_dir, exist_ok=True)
    out_model = os.path.join(art_dir, "best_model.joblib")

    print(f"[v0] Loading dataset: {in_path}")
    df = pd.read_csv(in_path, low_memory=False)

    if "Delivery_Time" not in df.columns:
        print("[v0] ERROR: 'Delivery_Time' target not found.")
        sys.exit(1)

    # Feature sets
    target = "Delivery_Time"
    numeric_features = [
        "Agent_Age", "Agent_Rating", "distance_km", "time_to_pickup_min", "order_hour", "order_wday"
    ]
    categorical_features = ["Weather", "Traffic", "Vehicle", "Area", "Category"]

    # Ensure engineered columns exist or fill with defaults
    for col in numeric_features:
        if col not in df.columns:
            df[col] = 0
    for col in categorical_features:
        if col not in df.columns:
            df[col] = "Unknown"

    X = df[numeric_features + categorical_features]
    y = df[target].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    candidates = []
    # Linear Regression
    candidates.append(("LinearRegression", LinearRegression()))
    # Random Forest
    candidates.append(("RandomForest", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)))
    # Gradient Boosting
    candidates.append(("GradientBoosting", GradientBoostingRegressor(random_state=42)))
    # XGBoost (optional)
    if HAS_XGB:
        candidates.append(("XGBoost", XGBRegressor(
            n_estimators=400, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, max_depth=6, random_state=42
        )))

    best_name = None
    best_score = np.inf
    best_pipeline = None
    all_results = []

    # MLflow setup
    if HAS_MLFLOW:
        mlruns_dir = os.path.join(base_dir, "mlruns")
        os.makedirs(mlruns_dir, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{mlruns_dir}")
        mlflow.set_experiment("delivery_time_prediction")

    for name, model in candidates:
        pipe = Pipeline(steps=[
            ("pre", preprocessor),
            ("model", model),
        ])
        if HAS_MLFLOW:
            with mlflow.start_run(run_name=name):
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                m = metrics_dict(y_test, preds)
                mlflow.log_params({"model": name})
                for k, v in m.items():
                    mlflow.log_metric(k, float(v))
                try:
                    mlflow.sklearn.log_model(pipe, artifact_path="model")
                except Exception:
                    pass
        else:
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            m = metrics_dict(y_test, preds)

        print(f"[v0] {name} metrics: RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} R2={m['r2']:.4f}")
        all_results.append({"model": name, **m})

        if m["rmse"] < best_score:
            best_score = m["rmse"]
            best_name = name
            best_pipeline = pipe

    if best_pipeline is None:
        print("[v0] ERROR: No model trained.")
        sys.exit(1)

    joblib.dump(best_pipeline, out_model)
    print(f"[v0] Saved best pipeline: {best_name} -> {out_model}")
    print("[v0] Summary:", json.dumps(all_results, indent=2))

if __name__ == "__main__":
    main()
