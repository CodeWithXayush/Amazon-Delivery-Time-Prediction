import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def save_fig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)

def main():
    # Prefer features.csv else processed.csv
    base_dir = os.path.dirname(__file__)
    default_feat = os.path.join(base_dir, "outputs", "features.csv")
    default_proc = os.path.join(base_dir, "outputs", "processed.csv")
    in_path = sys.argv[1] if len(sys.argv) > 1 else (default_feat if os.path.exists(default_feat) else default_proc)
    out_dir = os.path.join(base_dir, "outputs", "eda")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[v0] Reading data from: {in_path}")
    df = pd.read_csv(in_path, low_memory=False)

    # 1) Distribution of Delivery_Time
    if "Delivery_Time" in df.columns:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(df["Delivery_Time"].dropna(), bins=30, kde=True, ax=ax)
        ax.set_title("Distribution of Delivery Time (hours)")
        ax.set_xlabel("Delivery Time (hours)")
        save_fig(fig, os.path.join(out_dir, "delivery_time_distribution.png"))
        print("[v0] Saved: delivery_time_distribution.png")

    # 2) Distance vs Delivery Time
    if "distance_km" in df.columns and "Delivery_Time" in df.columns:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(x="distance_km", y="Delivery_Time", data=df, alpha=0.4, ax=ax)
        ax.set_title("Distance vs Delivery Time")
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Delivery Time (hours)")
        save_fig(fig, os.path.join(out_dir, "distance_vs_delivery_time.png"))
        print("[v0] Saved: distance_vs_delivery_time.png")

    # 3) Bar chart: Delivery Time by Category
    if "Category" in df.columns and "Delivery_Time" in df.columns:
        cat_agg = df.groupby("Category")["Delivery_Time"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(7,4))
        cat_agg.plot(kind="bar", ax=ax, color="#4B5563")
        ax.set_ylabel("Avg Delivery Time (hours)")
        ax.set_title("Average Delivery Time by Product Category")
        save_fig(fig, os.path.join(out_dir, "avg_delivery_time_by_category.png"))
        print("[v0] Saved: avg_delivery_time_by_category.png")

    # 4) Correlation heatmap of numeric features
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        corr = num_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(7,5))
        sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
        ax.set_title("Correlation Heatmap (numeric features)")
        save_fig(fig, os.path.join(out_dir, "correlation_heatmap.png"))
        print("[v0] Saved: correlation_heatmap.png")

    print("[v0] EDA completed.")

if __name__ == "__main__":
    main()
