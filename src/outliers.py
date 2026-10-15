"""
src/outliers.py
Module phân tích & xử lý giá trị ngoại lệ (Outliers) cho đồ án Customer Churn.

Quy trình 4 lớp:
  1. Phát hiện đơn biến  : IQR (1.5 & 3.0), Z-score, Modified Z-score (MAD)
  2. Phát hiện đa biến   : Isolation Forest, Local Outlier Factor, Mahalanobis (MCD)
  3. Kiểm tra logic       : TotalCharges có khớp tenure x MonthlyCharges hay không
  4. Xử lý                : OutlierCapper (fit trên train, transform trên test -> không rò rỉ dữ liệu)
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.config import FIGURES_DIR, RANDOM_STATE, REPORTS_DIR

OUTLIER_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "AvgChargePerMonth"]
MULTIVARIATE_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
OUTLIER_REPORT_DIR = REPORTS_DIR / "outlier_analysis"

sns.set_theme(style="whitegrid")


# ======================================================================
# 0. TIỆN ÍCH
# ======================================================================
def iqr_bounds(s: pd.Series, k: float = 1.5):
    """Trả về (lower, upper) theo luật hàng rào Tukey: [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def prepare_numeric_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo bảng số liệu sạch tối thiểu từ dữ liệu thô để phân tích outlier:
    ép TotalCharges về số (11 dòng tenure=0 điền bằng MonthlyCharges, đúng như pipeline chính)
    và thêm AvgChargePerMonth, nhãn y.
    """
    df = df_raw.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    df["AvgChargePerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)
    if "Churn" in df.columns:
        df["y"] = (df["Churn"].astype(str) == "Yes").astype(int)
    return df


# ======================================================================
# 1. PHÁT HIỆN ĐƠN BIẾN
# ======================================================================
def univariate_outlier_report(df: pd.DataFrame, cols=OUTLIER_COLUMNS) -> pd.DataFrame:
    """Bảng đếm số outlier của từng biến theo 4 quy tắc, kèm skewness và kurtosis."""
    rows = []
    for c in cols:
        x = df[c].astype(float)
        lo15, hi15 = iqr_bounds(x, 1.5)
        lo30, hi30 = iqr_bounds(x, 3.0)
        z = np.abs(stats.zscore(x))
        mad = stats.median_abs_deviation(x)
        mod_z = 0.6745 * (x - x.median()) / mad if mad > 0 else pd.Series(0.0, index=x.index)
        rows.append({
            "Biến": c,
            "Min": x.min(), "Max": x.max(),
            "Skewness": x.skew(), "Kurtosis": x.kurt(),
            "Hàng rào dưới (1.5 IQR)": lo15, "Hàng rào trên (1.5 IQR)": hi15,
            "IQR 1.5": int(((x < lo15) | (x > hi15)).sum()),
            "IQR 3.0 (cực đoan)": int(((x < lo30) | (x > hi30)).sum()),
            "Z-score > 3": int((z > 3).sum()),
            "Modified Z > 3.5 (MAD)": int((np.abs(mod_z) > 3.5).sum()),
        })
    return pd.DataFrame(rows).set_index("Biến")


# ======================================================================
# 2. PHÁT HIỆN ĐA BIẾN
# ======================================================================
def multivariate_outlier_flags(df: pd.DataFrame, cols=MULTIVARIATE_COLUMNS,
                               contamination: float = 0.01,
                               random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Gắn cờ outlier đa biến bằng 3 thuật toán độc lập (dữ liệu được chuẩn hóa trước):
      - iso  : Isolation Forest (contamination = 1%)
      - lof  : Local Outlier Factor, k = 30 (contamination = 1%)
      - maha : khoảng cách Mahalanobis robust (MCD), ngưỡng chi-square p = 0.999
    """
    X = StandardScaler().fit_transform(df[cols])
    flags = pd.DataFrame(index=df.index)

    iso = IsolationForest(n_estimators=300, contamination=contamination,
                          random_state=random_state).fit(X)
    flags["iso"] = iso.predict(X) == -1

    lof = LocalOutlierFactor(n_neighbors=30, contamination=contamination)
    flags["lof"] = lof.fit_predict(X) == -1

    mcd = MinCovDet(random_state=random_state, support_fraction=0.9).fit(X)
    flags["maha"] = mcd.mahalanobis(X) > stats.chi2.ppf(0.999, df=len(cols))

    flags["n_flags"] = flags[["iso", "lof", "maha"]].sum(axis=1)
    return flags


def profile_flagged_groups(df: pd.DataFrame, flags: pd.DataFrame,
                           cols=MULTIVARIATE_COLUMNS) -> pd.DataFrame:
    """So sánh nhóm bị gắn cờ với toàn bộ khách hàng (kích thước, tỷ lệ churn, giá trị trung bình)."""
    out = []
    groups = {"Toàn bộ khách hàng": pd.Series(True, index=df.index),
              "Isolation Forest": flags["iso"],
              "LOF": flags["lof"],
              "Mahalanobis (MCD)": flags["maha"],
              "Được >= 2 thuật toán gắn cờ": flags["n_flags"] >= 2}
    for name, mask in groups.items():
        sub = df[mask]
        row = {"Nhóm": name, "Số khách": len(sub), "Tỷ lệ churn": sub["y"].mean() if len(sub) else np.nan}
        for c in cols:
            row[f"{c} (TB)"] = sub[c].mean() if len(sub) else np.nan
        out.append(row)
    return pd.DataFrame(out).set_index("Nhóm")


# ======================================================================
# 3. KIỂM TRA TÍNH NHẤT QUÁN (OUTLIER LOGIC / LỖI NHẬP LIỆU)
# ======================================================================
def consistency_report(df: pd.DataFrame) -> dict:
    """
    Kiểm tra các điều kiện logic của dữ liệu cước phí:
      - tenure = 0  <=> chưa có hóa đơn
      - TotalCharges phải xấp xỉ tenure x MonthlyCharges (cho phép đổi gói cước)
      - Không có giá trị âm / trùng lặp
    """
    expected = df["tenure"] * df["MonthlyCharges"]
    ratio = (df["TotalCharges"] / expected.replace(0, np.nan)).dropna()
    return {
        "Số dòng trùng lặp": int(df.duplicated().sum()),
        "Số giá trị âm (tenure/MonthlyCharges/TotalCharges)": int((df[["tenure", "MonthlyCharges", "TotalCharges"]] < 0).sum().sum()),
        "Số dòng tenure = 0": int((df["tenure"] == 0).sum()),
        "tenure vượt 72 tháng": int((df["tenure"] > 72).sum()),
        "Tỷ lệ TotalCharges / (tenure x MonthlyCharges): min": float(ratio.min()),
        "Tỷ lệ ...: trung vị": float(ratio.median()),
        "Tỷ lệ ...: max": float(ratio.max()),
        "Số dòng tỷ lệ < 0.5 hoặc > 2 (nghi lỗi nhập liệu)": int(((ratio < 0.5) | (ratio > 2)).sum()),
        "Số dòng tỷ lệ ngoài hàng rào IQR (đổi gói cước, hợp lý)": int(
            ((ratio < iqr_bounds(ratio)[0]) | (ratio > iqr_bounds(ratio)[1])).sum()),
    }


# ======================================================================
# 4. XỬ LÝ: TRANSFORMER CHỐNG RÒ RỈ DỮ LIỆU
# ======================================================================
class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Capping (Winsorization) học ngưỡng trên tập TRAIN rồi áp lên mọi tập khác.

    method="iqr"        : ngưỡng = [Q1 - k*IQR, Q3 + k*IQR]
    method="percentile" : ngưỡng = [percentile thấp, percentile cao]  (ví dụ 1% / 99%)
    """

    def __init__(self, columns=OUTLIER_COLUMNS, method="iqr", k=1.5, lower_q=0.01, upper_q=0.99):
        self.columns = columns
        self.method = method
        self.k = k
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X: pd.DataFrame, y=None):
        self.bounds_ = {}
        for c in self.columns:
            if c not in X.columns:
                continue
            if self.method == "iqr":
                self.bounds_[c] = iqr_bounds(X[c], self.k)
            elif self.method == "percentile":
                self.bounds_[c] = (X[c].quantile(self.lower_q), X[c].quantile(self.upper_q))
            else:
                raise ValueError("method phải là 'iqr' hoặc 'percentile'")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        self.report_ = {}
        for c, (lo, hi) in self.bounds_.items():
            n_low, n_high = int((X[c] < lo).sum()), int((X[c] > hi).sum())
            self.report_[c] = {"lower": lo, "upper": hi, "n_below": n_low, "n_above": n_high}
            X[c] = X[c].clip(lower=lo, upper=hi)
        return X


# ======================================================================
# 5. BIỂU ĐỒ
# ======================================================================
LABELS = {"tenure": "Thời gian gắn bó (tháng)", "MonthlyCharges": "Phí hàng tháng ($)",
          "TotalCharges": "Tổng cước đã trả ($)", "AvgChargePerMonth": "Cước TB/tháng ($)"}


def plot_univariate_diagnostics(df: pd.DataFrame, cols=OUTLIER_COLUMNS, save_path=None):
    """Histogram + KDE kèm hàng rào IQR 1.5 và 3.0, boxplot phía dưới: nhìn thẳng được dữ liệu có outlier hay không."""
    save_path = save_path or FIGURES_DIR / "03b_outliers_detection.png"
    fig, axes = plt.subplots(2, len(cols), figsize=(4.6 * len(cols), 7),
                             gridspec_kw={"height_ratios": [3, 1.3]})
    for i, c in enumerate(cols):
        x = df[c]
        lo15, hi15 = iqr_bounds(x, 1.5)
        lo30, hi30 = iqr_bounds(x, 3.0)
        ax = axes[0, i]
        sns.histplot(x, kde=True, bins=40, color="#8e44ad", ax=ax, edgecolor="white")
        ax.axvline(x.min(), color="#7f8c8d", ls=":", lw=1.2)
        ax.axvline(x.max(), color="#7f8c8d", ls=":", lw=1.2, label=f"Min/Max: {x.min():.0f} – {x.max():.0f}")
        if hi15 < x.max() * 1.6:
            ax.axvline(hi15, color="#e67e22", ls="--", lw=1.6, label=f"Hàng rào trên 1.5·IQR = {hi15:.0f}")
        else:
            ax.plot([], [], color="#e67e22", ls="--", label=f"Hàng rào 1.5·IQR = [{lo15:.0f}; {hi15:.0f}]\n(nằm ngoài khung vẽ)")
        n15 = int(((x < lo15) | (x > hi15)).sum())
        n30 = int(((x < lo30) | (x > hi30)).sum())
        ax.set_title(f"{LABELS.get(c, c)}\nSố outlier: IQR1.5 = {n15} | IQR3.0 = {n30}", fontweight="bold", fontsize=10)
        ax.set_xlabel("")
        ax.legend(fontsize=7.5, loc="upper center")
        sns.boxplot(x=x, ax=axes[1, i], color="#c39bd3", width=0.5, whis=1.5,
                    medianprops=dict(color="yellow", linewidth=2), boxprops=dict(edgecolor="black"))
        axes[1, i].set_xlabel(c)
    fig.suptitle("Chẩn đoán Outlier đơn biến trên dữ liệu gốc (chưa xử lý)", fontweight="bold", fontsize=13, y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=130)
    plt.close()


def plot_multivariate_outliers(df: pd.DataFrame, flags: pd.DataFrame, save_path=None):
    """Scatter tenure vs MonthlyCharges, tô màu theo thuật toán gắn cờ -> thấy 3 thuật toán bắt 3 nhóm khác nhau."""
    save_path = save_path or FIGURES_DIR / "03c_multivariate_outliers.png"
    spec = [("iso", "Isolation Forest", "#e74c3c"),
            ("lof", "Local Outlier Factor", "#2980b9"),
            ("maha", "Mahalanobis (MCD)", "#27ae60")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)
    for ax, (key, name, color) in zip(axes, spec):
        ax.scatter(df["tenure"], df["MonthlyCharges"], s=6, color="#bdc3c7", alpha=0.5, label="Bình thường")
        m = flags[key]
        ax.scatter(df.loc[m, "tenure"], df.loc[m, "MonthlyCharges"], s=22, color=color, edgecolor="black",
                   linewidth=0.3, label=f"Bị gắn cờ (n={int(m.sum())}, churn={df.loc[m, 'y'].mean():.1%})")
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("tenure (tháng)")
        ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    axes[0].set_ylabel("MonthlyCharges ($)")
    fig.suptitle("Outlier đa biến: mỗi thuật toán bắt một nhóm khách hàng khác nhau", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=130)
    plt.close()


def plot_treatment_comparison(summary: pd.DataFrame, save_path=None):
    """Biểu đồ ROC-AUC trung bình ± độ lệch chuẩn theo phương pháp xử lý outlier, mỗi mô hình một panel."""
    save_path = save_path or FIGURES_DIR / "03d_outlier_treatment_comparison.png"
    models = summary.index.get_level_values("Mô hình").unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.6), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, models):
        sub = summary.loc[m].sort_values("AUC_mean")
        base = summary.loc[(m, "A. Không xử lý (baseline)"), "AUC_mean"]
        colors = ["#27ae60" if i == "A. Không xử lý (baseline)" else "#95a5a6" for i in sub.index]
        ax.barh(sub.index, sub["AUC_mean"], xerr=sub["AUC_std"], color=colors, edgecolor="black",
                linewidth=0.5, capsize=3)
        ax.axvline(base, color="#27ae60", ls="--", lw=1.2)
        lo = (sub["AUC_mean"] - sub["AUC_std"]).min()
        ax.set_xlim(lo - 0.004, (sub["AUC_mean"] + sub["AUC_std"]).max() + 0.004)
        ax.set_title(m, fontweight="bold")
        ax.set_xlabel("ROC-AUC trên tập test (TB ± SD qua các lần chia)")
    fig.suptitle("Ảnh hưởng của phương pháp xử lý outlier tới hiệu năng mô hình", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=130)
    plt.close()


# ======================================================================
# 6. THỰC NGHIỆM: XỬ LÝ OUTLIER ẢNH HƯỞNG THẾ NÀO TỚI MÔ HÌNH?
# ======================================================================
TREATMENTS = [
    "A. Không xử lý (baseline)",
    "B. Capping IQR 1.5 (cách đang dùng)",
    "C. Winsorize 1% - 99%",
    "D. Winsorize 5% - 95%",
    "E. Loại outlier Isolation Forest (train)",
    "F. Loại outlier LOF (train)",
    "G. log1p(TotalCharges)",
    "H. RobustScaler thay StandardScaler",
]


def _apply_treatment(name: str, X_tr: pd.DataFrame, X_te: pd.DataFrame, y_tr: pd.Series):
    """Áp một phương pháp xử lý: mọi tham số học trên TRAIN. Trả về (X_tr, X_te, y_tr, scaler_kind)."""
    scaler_kind = "standard"
    if name.startswith("B."):
        cap = OutlierCapper(method="iqr", k=1.5).fit(X_tr)
        X_tr, X_te = cap.transform(X_tr), cap.transform(X_te)
    elif name.startswith("C."):
        cap = OutlierCapper(method="percentile", lower_q=0.01, upper_q=0.99).fit(X_tr)
        X_tr, X_te = cap.transform(X_tr), cap.transform(X_te)
    elif name.startswith("D."):
        cap = OutlierCapper(method="percentile", lower_q=0.05, upper_q=0.95).fit(X_tr)
        X_tr, X_te = cap.transform(X_tr), cap.transform(X_te)
    elif name.startswith("E.") or name.startswith("F."):
        Z = StandardScaler().fit_transform(X_tr[MULTIVARIATE_COLUMNS])
        if name.startswith("E."):
            bad = IsolationForest(n_estimators=300, contamination=0.01, random_state=RANDOM_STATE).fit_predict(Z) == -1
        else:
            bad = LocalOutlierFactor(n_neighbors=30, contamination=0.01).fit_predict(Z) == -1
        X_tr, y_tr = X_tr.loc[~bad], y_tr.loc[~bad]
    elif name.startswith("G."):
        X_tr, X_te = X_tr.copy(), X_te.copy()
        X_tr["TotalCharges"], X_te["TotalCharges"] = np.log1p(X_tr["TotalCharges"]), np.log1p(X_te["TotalCharges"])
    elif name.startswith("H."):
        scaler_kind = "robust"
    return X_tr, X_te, y_tr, scaler_kind


def run_treatment_experiment(df_raw: pd.DataFrame, n_splits: int = 10, treatments=TREATMENTS,
                             model_names=("Logistic Regression", "Random Forest", "Gradient Boosting")):
    """
    Lặp n_splits lần chia train/test (stratified, seed khác nhau). Mỗi lần:
      xử lý outlier (học trên train) -> scale -> huấn luyện -> chấm điểm trên test NGUYÊN VẸN.
    Trả về (raw_results, summary). `summary` có AUC/Recall/F1 trung bình ± SD và chênh lệch ghép cặp so với baseline.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import roc_auc_score, recall_score, f1_score
    from src.config import ID_COLUMN, NUMERICAL_COLUMNS, TARGET_COLUMN, TEST_SIZE
    from src.modeling import get_model_zoo
    from src.preprocessing import (handle_missing_data, clean_categorical_values,
                                   create_features, encode_features)

    feat = encode_features(create_features(clean_categorical_values(handle_missing_data(df_raw))))
    X_all, y_all = feat.drop(columns=[TARGET_COLUMN]), feat[TARGET_COLUMN]
    scale_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges", "AvgChargePerMonth", "NumServices"] if c in X_all]

    rows = []
    for split in range(n_splits):
        seed = RANDOM_STATE + split
        X_tr0, X_te0, y_tr0, y_te = train_test_split(X_all, y_all, test_size=TEST_SIZE,
                                                     random_state=seed, stratify=y_all)
        for t in treatments:
            X_tr, X_te, y_tr, kind = _apply_treatment(t, X_tr0, X_te0, y_tr0)
            sc = RobustScaler() if kind == "robust" else StandardScaler()
            X_tr, X_te = X_tr.copy(), X_te.copy()
            X_tr[scale_cols] = sc.fit_transform(X_tr[scale_cols])
            X_te[scale_cols] = sc.transform(X_te[scale_cols])
            zoo = get_model_zoo()
            for m in model_names:
                model = zoo[m]
                if hasattr(model, "random_state"):
                    model.set_params(random_state=seed)
                model.fit(X_tr, y_tr)
                prob = model.predict_proba(X_te)[:, 1]
                pred = (prob >= 0.5).astype(int)
                rows.append({"split": split, "Mô hình": m, "Phương pháp": t, "n_train": len(X_tr),
                             "AUC": roc_auc_score(y_te, prob), "Recall": recall_score(y_te, pred),
                             "F1": f1_score(y_te, pred)})
    raw = pd.DataFrame(rows)

    base = raw[raw["Phương pháp"] == treatments[0]].set_index(["Mô hình", "split"])["AUC"]
    out = []
    for (m, t), g in raw.groupby(["Mô hình", "Phương pháp"]):
        diff = g.set_index("split")["AUC"] - base.loc[m]
        p = stats.ttest_1samp(diff, 0.0).pvalue if diff.abs().sum() > 0 else 1.0
        out.append({"Mô hình": m, "Phương pháp": t, "n_train (TB)": g["n_train"].mean(),
                    "AUC_mean": g["AUC"].mean(), "AUC_std": g["AUC"].std(),
                    "Recall_mean": g["Recall"].mean(), "F1_mean": g["F1"].mean(),
                    "ΔAUC vs baseline": diff.mean(), "p-value (ghép cặp)": p})
    summary = pd.DataFrame(out).set_index(["Mô hình", "Phương pháp"])
    return raw, summary
