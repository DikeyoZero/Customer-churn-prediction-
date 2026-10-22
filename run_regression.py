"""
run_regression.py
Chạy toàn bộ phần Hồi quy tuyến tính (độc lập với main.py):
    python run_regression.py
Kết quả: reports/regression_analysis/*.csv|json  và  reports/figures/13..16_reg_*.png
"""
import json
import time

import pandas as pd

from src.config import FIGURES_DIR
from src.regression import (REGRESSION_REPORT_DIR, coefficient_table, diagnostics_report, fit_ols,
                            honest_evaluation, linear_probability_demo, load_regression_data,
                            overfitting_demo, plot_diagnostics, plot_lpm_misuse, plot_overfitting,
                            plot_regularization_paths, regularization_full, split_regression_data,
                            train_test_gap_by_n, vif_with_numservices)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def main():
    t0 = time.time()
    REGRESSION_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df, X, y = load_regression_data()
    X_train, X_test, y_train, y_test = split_regression_data(X, y)
    print(f"[*] Train {len(X_train):,} | Test {len(X_test):,} | {X.shape[1]} biến | mục tiêu = MonthlyCharges")

    print("\n[1] Mô hình OLS - bảng hệ số")
    ols = fit_ols(X_train, y_train)
    coef = coefficient_table(ols)
    coef.round(4).to_csv(REGRESSION_REPORT_DIR / "01_coefficients.csv")
    print(coef.round(4))

    print("\n[2] Đánh giá trung thực")
    ev = honest_evaluation(X_train, X_test, y_train, y_test)
    ev.round(4).to_csv(REGRESSION_REPORT_DIR / "02_evaluation.csv")
    print(ev.round(4))

    print("\n[3] Chẩn đoán 4 giả định")
    diag = diagnostics_report(ols)
    diag["vif_max_with_NumServices"] = vif_with_numservices(X_train, df)
    lpm_info, p, lpm_ols = linear_probability_demo(df)
    diag["linear_probability_churn"] = lpm_info
    json.dump(diag, open(REGRESSION_REPORT_DIR / "03_diagnostics.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    plot_diagnostics(ols)
    plot_lpm_misuse(p, lpm_ols)
    print(json.dumps(diag, indent=1, ensure_ascii=False))

    print("\n[4] Overfitting & điều chuẩn")
    demo = overfitting_demo(X_train, y_train, X_test, y_test)
    gap = train_test_gap_by_n(X_train, y_train, X_test, y_test)
    demo.round(4).to_csv(REGRESSION_REPORT_DIR / "04_overfitting_demo.csv")
    gap.round(4).to_csv(REGRESSION_REPORT_DIR / "04_train_test_gap_by_n.csv")
    plot_overfitting(demo, gap)
    table, lasso_coef, paths = regularization_full(X_train, X_test, y_train, y_test)
    table.round(4).to_csv(REGRESSION_REPORT_DIR / "05_regularization_full.csv")
    plot_regularization_paths(paths)
    print(demo.round(4)); print(gap.round(4)); print(table.round(4))
    print("Lasso ép về 0:", list(lasso_coef[lasso_coef == 0].index))

    print(f"\n[+] Xong trong {time.time() - t0:.1f}s. Hình: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
