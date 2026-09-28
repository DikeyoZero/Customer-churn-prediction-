"""
src/eda.py
Module trực quan hóa và khám phá dữ liệu (Exploratory Data Analysis).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import FIGURES_DIR

# Cấu hình thẩm mỹ cho biểu đồ
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 10


def plot_churn_distribution(df: pd.DataFrame, save_path=None):
    """Vẽ biểu đồ phân bố nhãn Churn (Pie Chart & Bar Chart)."""
    if save_path is None:
        save_path = FIGURES_DIR / "01_churn_distribution.png"

    churn_counts = df["Churn"].value_counts()
    colors = ["#2ecc71", "#e74c3c"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # 1. Pie Chart
    wedges, texts, autotexts = axes[0].pie(
        churn_counts, labels=["No Churn", "Churn"],
        autopct="%1.1f%%", colors=colors, startangle=90
    )
    for at in autotexts:
        at.set_fontweight("bold")
    axes[0].set_title("Tỷ lệ Khách hàng Rời bỏ (Imbalance Ratio)", fontweight="bold")

    # 2. Bar Chart
    bars = axes[1].bar(["No Churn", "Churn"], churn_counts.values, color=colors, edgecolor="black", width=0.5)
    axes[1].set_title("Số lượng Khách hàng theo Nhãn", fontweight="bold")
    axes[1].set_ylabel("Số lượng khách hàng")
    for bar, val in zip(bars, churn_counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50, f"{val:,}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_categorical_insights(df: pd.DataFrame, save_path=None):
    """Vẽ tỷ lệ Churn theo các nhóm thuộc tính phân loại cốt lõi."""
    if save_path is None:
        save_path = FIGURES_DIR / "02_churn_by_categorical.png"

    cat_cols = [
        ("Contract", "Loại Hợp đồng"),
        ("InternetService", "Dịch vụ Internet"),
        ("PaymentMethod", "Phương thức Thanh toán"),
        ("TechSupport", "Hỗ trợ Kỹ thuật")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    avg_churn = (df["Churn"] == "Yes").mean() * 100

    for i, (col, title) in enumerate(cat_cols):
        churn_rate = df.groupby(col)["Churn"].apply(
            lambda x: (x == "Yes").mean() * 100
        ).sort_values(ascending=False)

        colors = ["#e74c3c" if v > avg_churn else "#2ecc71" for v in churn_rate.values]
        bars = axes[i].bar(churn_rate.index, churn_rate.values, color=colors, edgecolor="black", width=0.6)
        axes[i].axhline(avg_churn, color="navy", linestyle="--", linewidth=1.2, label=f"Churn TB: {avg_churn:.1f}%")
        axes[i].set_title(f"Tỷ lệ Churn theo {title}", fontweight="bold")
        axes[i].set_ylabel("Tỷ lệ Churn (%)")
        axes[i].tick_params(axis="x", rotation=15)
        axes[i].legend(loc="upper right", fontsize=8)

        for bar, val in zip(bars, churn_rate.values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_numerical_distributions(df: pd.DataFrame, save_path=None):
    """Vẽ phân phối và Boxplot cho các biến số (tenure, MonthlyCharges, TotalCharges)."""
    if save_path is None:
        save_path = FIGURES_DIR / "03_numerical_distributions.png"

    df_plot = df.copy()
    df_plot["TotalCharges"] = pd.to_numeric(df_plot["TotalCharges"], errors="coerce").fillna(0)

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    col_names = ["Thời gian gắn bó (tháng)", "Phí hàng tháng ($)", "Tổng cước đã trả ($)"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for i, (col, name) in enumerate(zip(num_cols, col_names)):
        # Histogram
        axes[0, i].hist(df_plot[col], bins=30, color="#3498db", edgecolor="white", alpha=0.8)
        axes[0, i].set_title(f"Phân phối {name}", fontweight="bold")
        axes[0, i].set_ylabel("Tần suất")

        # Boxplot theo Churn (chú ý dùng tick_labels tương thích matplotlib mới)
        churn_no = df_plot[df_plot["Churn"] == "No"][col]
        churn_yes = df_plot[df_plot["Churn"] == "Yes"][col]
        axes[1, i].boxplot([churn_no, churn_yes], tick_labels=["No Churn", "Churn"], patch_artist=True,
                           boxprops=dict(facecolor="#2ecc71" if i == 0 else "#e74c3c"),
                           medianprops=dict(color="black", linewidth=2))
        axes[1, i].set_title(f"{name} theo Churn", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def generate_all_eda_plots(df: pd.DataFrame):
    """Chạy toàn bộ quy trình trực quan hóa dữ liệu và xuất biểu đồ."""
    plot_churn_distribution(df)
    plot_categorical_insights(df)
    plot_numerical_distributions(df)
