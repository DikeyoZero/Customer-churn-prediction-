"""
src/regression.py
Module hồi quy tuyến tính cho đồ án Customer Churn (dữ liệu Telco).

Bốn mục tiêu:
  1. Xây dựng & đọc mô hình : OLS dự đoán MonthlyCharges, diễn giải từng hệ số
  2. Đánh giá trung thực    : baseline, hold-out (test chưa từng thấy), K-fold CV
  3. Chẩn đoán mô hình      : 4 giả định LINE + VIF + Cook's distance; ví dụ KHÔNG nên dùng (Churn)
  4. Kiểm soát overfitting  : tách dữ liệu, CV, Ridge (L2), Lasso (L1)

Lưu ý: biến mục tiêu là MonthlyCharges (liên tục). TotalCharges, AvgChargePerMonth bị loại vì
được tính từ chính MonthlyCharges (rò rỉ dữ liệu); NumServices bị loại vì bằng tổng các cột dịch vụ
(đa cộng tuyến hoàn hảo).
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.config import CV_FOLDS, FIGURES_DIR, PROCESSED_DATA_PATH, RANDOM_STATE, REPORTS_DIR, TEST_SIZE

REG_TARGET = "MonthlyCharges"
REG_EXCLUDED = ["MonthlyCharges", "TotalCharges", "AvgChargePerMonth", "NumServices", "Churn"]
REGRESSION_REPORT_DIR = REPORTS_DIR / "regression_analysis"


# ---------------------------------------------------------------------------
# 0. DỮ LIỆU
# ---------------------------------------------------------------------------
def load_regression_data(path=PROCESSED_DATA_PATH):
    """Đọc telco_cleaned.csv, trả về (df đầy đủ, X, y) cho bài hồi quy MonthlyCharges."""
    df = pd.read_csv(path)
    feats = [c for c in df.columns if c not in REG_EXCLUDED]
    return df, df[feats].astype(float), df[REG_TARGET]


def split_regression_data(X, y):
    """Chia train/test đúng cấu hình config.py (80/20, random_state=42)."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def _metrics(y_true, y_pred) -> dict:
    return {"R2": r2_score(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
            "MAE": mean_absolute_error(y_true, y_pred)}


# ---------------------------------------------------------------------------
# 1. XÂY DỰNG & ĐỌC MÔ HÌNH
# ---------------------------------------------------------------------------
def fit_ols(X_train, y_train):
    """Fit OLS bằng statsmodels (có hằng số) để có hệ số, khoảng tin cậy và p-value."""
    return sm.OLS(y_train, sm.add_constant(X_train)).fit()


def coefficient_table(ols) -> pd.DataFrame:
    """Bảng hệ số: coef, khoảng tin cậy 95%, p-value, cờ 'có ý nghĩa' (p < 0.05)."""
    ci = ols.conf_int()
    t = pd.DataFrame({"coef": ols.params, "ci_low": ci[0], "ci_high": ci[1], "p_value": ols.pvalues})
    t["significant"] = t["p_value"] < 0.05
    return t


# ---------------------------------------------------------------------------
# 2. ĐÁNH GIÁ TRUNG THỰC
# ---------------------------------------------------------------------------
def honest_evaluation(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Baseline (đoán trung bình) vs OLS trên train / test / CV K-fold."""
    lr = LinearRegression().fit(X_train, y_train)
    cv = cross_validate(LinearRegression(), X_train, y_train,
                        cv=KFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                        scoring=["r2", "neg_root_mean_squared_error"])
    rows = {
        "Baseline (đoán trung bình train) - test": _metrics(y_test, np.full(len(y_test), y_train.mean())),
        "OLS - train": _metrics(y_train, lr.predict(X_train)),
        "OLS - test (chưa từng thấy)": _metrics(y_test, lr.predict(X_test)),
    }
    out = pd.DataFrame(rows).T
    out.loc[f"OLS - CV {CV_FOLDS}-fold (trung bình)"] = [cv["test_r2"].mean(),
                                                         -cv["test_neg_root_mean_squared_error"].mean(), np.nan]
    out["R2_std_CV"] = np.nan
    out.loc[f"OLS - CV {CV_FOLDS}-fold (trung bình)", "R2_std_CV"] = cv["test_r2"].std()
    return out


# ---------------------------------------------------------------------------
# 3. CHẨN ĐOÁN MÔ HÌNH
# ---------------------------------------------------------------------------
def diagnostics_report(ols) -> dict:
    """Kiểm định 4 giả định LINE + VIF + Cook's distance."""
    X = ols.model.exog
    res, fit = ols.resid, ols.fittedvalues
    bp = het_breuschpagan(res, X)
    reset = linear_reset(ols, power=2, use_f=True)
    jb = jarque_bera(res)
    sh = stats.shapiro(res.sample(min(4000, len(res)), random_state=1))
    names = list(ols.model.exog_names)[1:]
    vif = pd.Series([variance_inflation_factor(X, i) for i in range(1, X.shape[1])], index=names)
    cook = ols.get_influence().cooks_distance[0]
    tercile = pd.qcut(fit, 3, labels=["thấp", "giữa", "cao"])
    return {
        "linearity_RESET": {"F": float(reset.fvalue), "p": float(reset.pvalue)},
        "independence_DW": float(durbin_watson(res)),
        "normality_JB": {"stat": float(jb[0]), "p": float(jb[1]), "skew": float(jb[2]), "kurtosis": float(jb[3])},
        "normality_Shapiro": {"W": float(sh[0]), "p": float(sh[1])},
        "homoscedasticity_BP": {"LM": float(bp[0]), "p": float(bp[1])},
        "resid_sd_by_fitted_tercile": res.groupby(tercile).std().round(3).to_dict(),
        "vif": vif.sort_values(ascending=False).round(2).to_dict(),
        "cooks_distance_max": float(cook.max()),
    }


def vif_with_numservices(X_train, df) -> float:
    """Minh họa đa cộng tuyến hoàn hảo: thêm NumServices (= tổng các cột dịch vụ) -> VIF cực lớn."""
    X2 = sm.add_constant(pd.concat([X_train, df.loc[X_train.index, "NumServices"].astype(float)], axis=1))
    with warnings.catch_warnings():  # cảnh báo ma trận suy biến là đúng ý minh họa
        warnings.simplefilter("ignore")
        return float(max(variance_inflation_factor(X2.values, i) for i in range(1, X2.shape[1])))


def plot_diagnostics(ols, save_path=None):
    """4 biểu đồ chẩn đoán: Residuals vs Fitted, Q-Q, Scale-Location, phần dư theo thứ tự."""
    save_path = save_path or FIGURES_DIR / "13_reg_diagnostics.png"
    res, fit = ols.resid, ols.fittedvalues
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].scatter(fit, res, s=5, alpha=.35, color="#2a6f97"); ax[0, 0].axhline(0, color="r", lw=1)
    ax[0, 0].set(title="1) Residuals vs Fitted (tuyến tính)", xlabel="Giá trị dự đoán", ylabel="Phần dư")
    sm.qqplot(res, line="s", ax=ax[0, 1], markersize=3); ax[0, 1].set_title("2) Q-Q plot (phần dư chuẩn)")
    ax[1, 0].scatter(fit, np.sqrt(np.abs(res / res.std())), s=5, alpha=.35, color="#2a6f97")
    ax[1, 0].set(title="3) Scale-Location (đồng phương sai)", xlabel="Giá trị dự đoán", ylabel="√|phần dư chuẩn hóa|")
    ax[1, 1].plot(res.sort_index().values[:300], lw=.7, color="#2a6f97"); ax[1, 1].axhline(0, color="r", lw=1)
    ax[1, 1].set(title="4) Phần dư theo thứ tự (độc lập)", xlabel="Thứ tự", ylabel="Phần dư")
    plt.tight_layout(); plt.savefig(save_path, dpi=130, bbox_inches="tight"); plt.close()


def linear_probability_demo(df):
    """Ví dụ KHÔNG nên dùng: hồi quy tuyến tính dự đoán Churn (0/1) so với Logistic Regression."""
    cols = [c for c in df.columns if c not in ["Churn", "TotalCharges", "AvgChargePerMonth", "NumServices"]]
    X, y = df[cols].astype(float), df["Churn"]
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    lpm = LinearRegression().fit(Xa, ya); p = lpm.predict(Xb)
    sc = StandardScaler().fit(Xa)
    lg = LogisticRegression(max_iter=2000).fit(sc.transform(Xa), ya)
    ols = sm.OLS(ya, sm.add_constant(Xa)).fit()
    info = {"n_test": len(p), "pred_below_0": int((p < 0).sum()), "pred_above_1": int((p > 1).sum()),
            "pred_min": float(p.min()), "pred_max": float(p.max()), "R2_test": r2_score(yb, p),
            "AUC_linear": roc_auc_score(yb, p), "AUC_logistic": roc_auc_score(yb, lg.predict_proba(sc.transform(Xb))[:, 1]),
            "BP_p": float(het_breuschpagan(ols.resid, sm.add_constant(Xa))[1]),
            "Shapiro_p": float(stats.shapiro(ols.resid.sample(4000, random_state=1))[1])}
    return info, p, ols


def plot_lpm_misuse(p, ols, save_path=None):
    save_path = save_path or FIGURES_DIR / "16_reg_lpm_misuse.png"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(p, bins=40, color="#577590"); ax[0].axvline(0, color="r"); ax[0].axvline(1, color="r")
    ax[0].set(title="Hồi quy tuyến tính dự đoán Churn: xác suất ra ngoài [0,1]", xlabel="Giá trị dự đoán")
    ax[1].scatter(ols.fittedvalues, ols.resid, s=4, alpha=.3, color="#577590")
    ax[1].set(title="Phần dư tạo 2 đường song song (Churn chỉ 0/1)", xlabel="Dự đoán", ylabel="Phần dư")
    plt.tight_layout(); plt.savefig(save_path, dpi=130, bbox_inches="tight"); plt.close()


# ---------------------------------------------------------------------------
# 4. KIỂM SOÁT OVERFITTING
# ---------------------------------------------------------------------------
def overfitting_demo(X_train, y_train, X_test, y_test, n_small=150, seed=0):
    """Cố ý gây overfitting: n_small dòng train + đặc trưng bậc 2 (252 cột). So OLS bậc 1/2, Ridge, Lasso."""
    idx = np.random.RandomState(seed).choice(len(X_train), n_small, replace=False)
    Xs, ys = X_train.iloc[idx], y_train.iloc[idx]
    poly = lambda: PolynomialFeatures(2, include_bias=False)
    makers = {
        "OLS bậc 1": lambda: make_pipeline(StandardScaler(), LinearRegression()),
        "OLS bậc 2": lambda: make_pipeline(poly(), StandardScaler(), LinearRegression()),
        "Ridge bậc 2 (α theo CV)": lambda: make_pipeline(poly(), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30))),
        "Lasso bậc 2 (α theo CV)": lambda: make_pipeline(poly(), StandardScaler(),
                                                         LassoCV(alphas=np.logspace(-3, 1, 30), cv=5, max_iter=20000)),
    }
    rows = []
    for name, mk in makers.items():
        m = mk().fit(Xs, ys)
        cv = cross_val_score(mk(), Xs, ys, cv=KFold(5, shuffle=True, random_state=seed), scoring="r2").mean()
        rows.append({"Mô hình": name, "R2 train": r2_score(ys, m.predict(Xs)), "R2 CV 5-fold": cv,
                     "R2 test": r2_score(y_test, m.predict(X_test)),
                     "RMSE test": mean_squared_error(y_test, m.predict(X_test)) ** 0.5})
    return pd.DataFrame(rows).set_index("Mô hình")


def train_test_gap_by_n(X_train, y_train, X_test, y_test, sizes=(100, 200, 400, 800, 1600, 4000)):
    """OLS bậc 2: khoảng cách train-test thu hẹp khi tăng số mẫu."""
    out = []
    for n in sizes:
        idx = np.random.RandomState(1).choice(len(X_train), n, replace=False)
        m = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), LinearRegression())
        m.fit(X_train.iloc[idx], y_train.iloc[idx])
        out.append({"n_train": n, "R2 train": r2_score(y_train.iloc[idx], m.predict(X_train.iloc[idx])),
                    "R2 test": r2_score(y_test, m.predict(X_test))})
    return pd.DataFrame(out).set_index("n_train")


def plot_overfitting(demo, gap, save_path=None):
    save_path = save_path or FIGURES_DIR / "14_reg_overfitting.png"
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    v = demo["RMSE test"]
    ax[0].bar(v.index, v.values, color=["#90be6d", "#f94144", "#f9c74f", "#43aa8b"])
    for i, x in enumerate(v.values): ax[0].text(i, x + .1, f"{x:.2f}", ha="center")
    ax[0].set(title="RMSE trên tập test (n_train=150, 252 đặc trưng bậc 2)", ylabel="RMSE (USD)")
    ax[0].tick_params(axis="x", labelsize=8)
    ax[1].plot(gap.index, gap["R2 train"], "o-", label="Train R²", color="#90be6d")
    ax[1].plot(gap.index, gap["R2 test"], "o-", label="Test R²", color="#f94144")
    ax[1].set_xscale("log"); ax[1].legend()
    ax[1].set(title="OLS bậc 2: khoảng cách Train–Test thu hẹp khi n tăng", xlabel="Số mẫu train (log)", ylabel="R²")
    plt.tight_layout(); plt.savefig(save_path, dpi=130, bbox_inches="tight"); plt.close()


def regularization_full(X_train, X_test, y_train, y_test):
    """Ridge/Lasso trên toàn bộ train (đã chuẩn hóa, scaler fit trên train). Trả về bảng + hệ số + đường đi."""
    feats = list(X_train.columns)
    sc = StandardScaler().fit(X_train)
    Xtr, Xte = sc.transform(X_train), sc.transform(X_test)
    ols = LinearRegression().fit(Xtr, y_train)
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5).fit(Xtr, y_train)
    lasso = LassoCV(alphas=np.logspace(-3, 1, 50), cv=5, max_iter=50000).fit(Xtr, y_train)
    table = pd.DataFrame({
        "alpha": [np.nan, ridge.alpha_, lasso.alpha_],
        "RMSE test": [_metrics(y_test, m.predict(Xte))["RMSE"] for m in (ols, ridge, lasso)],
        "Số hệ số = 0": [0, 0, int((lasso.coef_ == 0).sum())],
    }, index=["OLS", "Ridge", "Lasso"])
    paths = {
        "ridge_alphas": np.logspace(-3, 3, 50),
        "ridge": np.array([Ridge(alpha=a).fit(Xtr, y_train).coef_ for a in np.logspace(-3, 3, 50)]),
        "lasso_alphas": np.logspace(-3, 1, 50),
        "lasso": np.array([Lasso(alpha=a, max_iter=50000).fit(Xtr, y_train).coef_ for a in np.logspace(-3, 1, 50)]),
        "ridge_alpha": ridge.alpha_, "lasso_alpha": lasso.alpha_,
    }
    return table, pd.Series(lasso.coef_, index=feats), paths


def plot_regularization_paths(paths, save_path=None):
    save_path = save_path or FIGURES_DIR / "15_reg_regularization_path.png"
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(paths["ridge_alphas"], paths["ridge"]); ax[0].set_xscale("log")
    ax[0].axvline(paths["ridge_alpha"], color="k", ls="--")
    ax[0].set(title="Ridge: hệ số co dần về 0", xlabel="α (log)", ylabel="Hệ số (đã chuẩn hóa)")
    ax[1].plot(paths["lasso_alphas"], paths["lasso"]); ax[1].set_xscale("log")
    ax[1].axvline(paths["lasso_alpha"], color="k", ls="--")
    ax[1].set(title="Lasso: hệ số bị ép về đúng 0", xlabel="α (log)")
    plt.tight_layout(); plt.savefig(save_path, dpi=130, bbox_inches="tight"); plt.close()
