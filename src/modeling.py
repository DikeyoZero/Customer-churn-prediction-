"""
src/modeling.py
Module xây dựng, huấn luyện và đánh giá các mô hình Machine Learning.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)

from src.config import RANDOM_STATE, CV_FOLDS, FIGURES_DIR


def get_model_zoo() -> dict:
    """Khởi tạo danh sách các thuật toán phân loại cần so sánh."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=1000,
            C=1.0
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_STATE
        )
    }


def evaluate_model(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    """Huấn luyện mô hình và tính toán đầy đủ các chỉ số đánh giá."""
    # 1. Huấn luyện
    model.fit(X_train, y_train)

    # 2. Dự báo
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 3. Tính toán Metrics
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    # 4. Cross-Validation (K-Fold với phân tầng)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()

    return {
        "Model": name,
        "Accuracy": acc,
        "ROC-AUC": roc,
        "F1-Score": f1,
        "Precision": prec,
        "Recall": rec,
        "CV-ROC-AUC": cv_auc,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "model_obj": model
    }


def plot_model_comparison(results: dict, y_test, save_path=None):
    """Vẽ biểu đồ so sánh ROC-AUC, F1, Recall và đường cong ROC giữa các mô hình."""
    if save_path is None:
        save_path = FIGURES_DIR / "04_model_comparison.png"

    df_res = pd.DataFrame(results).T.drop(columns=["Model", "y_prob", "y_pred", "model_obj"]).astype(float)
    palette = ["#3498db", "#e67e22", "#27ae60", "#8e44ad"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Bar Chart so sánh ROC-AUC & F1-Score
    x = np.arange(len(df_res))
    width = 0.35

    axes[0].bar(x - width/2, df_res["ROC-AUC"], width, label="ROC-AUC", color="#27ae60")
    axes[0].bar(x + width/2, df_res["F1-Score"], width, label="F1-Score", color="#3498db")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_res.index, rotation=15)
    axes[0].set_ylim(0.4, 0.95)
    axes[0].set_title("So sánh ROC-AUC và F1-Score giữa các Mô hình", fontweight="bold")
    axes[0].legend(loc="upper right")

    # 2. ROC Curves
    for (m_name, m_data), color in zip(results.items(), palette):
        fpr, tpr, _ = roc_curve(y_test, m_data["y_prob"])
        auc_score = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, label=f"{m_name} (AUC={auc_score:.3f})", color=color, linewidth=2)

    axes[1].plot([0, 1], [0, 1], "k--", label="Random Classifier")
    axes[1].set_title("Đường cong ROC (Receiver Operating Characteristic)", fontweight="bold")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate (Recall)")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_feature_importance(rf_model, feature_names, save_path=None):
    """Vẽ Top 15 đặc trưng quan trọng nhất từ mô hình Random Forest."""
    if save_path is None:
        save_path = FIGURES_DIR / "05_feature_importance.png"

    importances = pd.Series(rf_model.feature_importances_, index=feature_names)
    top15 = importances.sort_values(ascending=True).tail(15)

    plt.figure(figsize=(9, 6))
    top15.plot(kind="barh", color="#27ae60", edgecolor="black", linewidth=0.6)
    plt.title("Top 15 Đặc trưng Quan trọng nhất (Random Forest Feature Importance)", fontweight="bold")
    plt.xlabel("Mức độ quan trọng (Feature Importance Score)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def run_modeling_pipeline(X_train, X_test, y_train, y_test):
    """Quy trình huấn luyện, đánh giá và xuất báo cáo tất cả các mô hình."""
    models = get_model_zoo()
    results = {}

    for name, model in models.items():
        res = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        results[name] = res

    # Vẽ biểu đồ so sánh
    plot_model_comparison(results, y_test)

    # Vẽ Feature Importance nếu có Random Forest
    if "Random Forest" in results:
        rf_obj = results["Random Forest"]["model_obj"]
        plot_feature_importance(rf_obj, X_train.columns)

    return results
