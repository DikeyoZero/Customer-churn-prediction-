"""
run_outliers.py
Chạy toàn bộ phần phân tích Outlier (độc lập với main.py):
    python run_outliers.py              # đầy đủ, gồm thực nghiệm 10 lần chia dữ liệu
    python run_outliers.py --splits 3   # chạy nhanh hơn
    python run_outliers.py --skip-experiment
Kết quả: reports/outlier_analysis/*.csv  và  reports/figures/03b..03d_*.png
"""
import argparse
import json
import time

import pandas as pd

from src.config import FIGURES_DIR
from src.outliers import (OUTLIER_REPORT_DIR, consistency_report, multivariate_outlier_flags,
                          plot_multivariate_outliers, plot_treatment_comparison,
                          plot_univariate_diagnostics, prepare_numeric_frame,
                          profile_flagged_groups, run_treatment_experiment,
                          univariate_outlier_report)
from src.preprocessing import load_raw_data

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def main(n_splits: int, skip_experiment: bool):
    t0 = time.time()
    OUTLIER_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df_raw = load_raw_data()
    df = prepare_numeric_frame(df_raw)

    print("=== 1. OUTLIER ĐƠN BIẾN (dữ liệu gốc) ===")
    uni = univariate_outlier_report(df)
    print(uni.round(3).T.to_string())
    uni.to_csv(OUTLIER_REPORT_DIR / "01_univariate_report.csv", encoding="utf-8-sig")
    plot_univariate_diagnostics(df)

    print("\n=== 2. OUTLIER ĐA BIẾN ===")
    flags = multivariate_outlier_flags(df)
    prof = profile_flagged_groups(df, flags)
    print(prof.round(3).to_string())
    prof.to_csv(OUTLIER_REPORT_DIR / "02_multivariate_profile.csv", encoding="utf-8-sig")
    out = df.loc[flags["n_flags"] >= 1, ["tenure", "MonthlyCharges", "TotalCharges", "Contract",
                                         "InternetService", "Churn"]].join(flags)
    out.to_csv(OUTLIER_REPORT_DIR / "02_flagged_customers.csv", encoding="utf-8-sig")
    plot_multivariate_outliers(df, flags)

    print("\n=== 3. KIỂM TRA NHẤT QUÁN ===")
    cons = consistency_report(df)
    for k, v in cons.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    (OUTLIER_REPORT_DIR / "03_consistency_report.json").write_text(
        json.dumps(cons, ensure_ascii=False, indent=2), encoding="utf-8")

    if not skip_experiment:
        print(f"\n=== 4. THỰC NGHIỆM XỬ LÝ OUTLIER ({n_splits} lần chia train/test) ===")
        raw, summary = run_treatment_experiment(df_raw, n_splits=n_splits)
        raw.to_csv(OUTLIER_REPORT_DIR / "04_experiment_raw.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(OUTLIER_REPORT_DIR / "04_experiment_summary.csv", encoding="utf-8-sig")
        print(summary.round(4).to_string())
        plot_treatment_comparison(summary)

    print(f"\nHoàn tất trong {time.time() - t0:.1f}s. Kết quả: {OUTLIER_REPORT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--skip-experiment", action="store_true")
    a = ap.parse_args()
    main(a.splits, a.skip_experiment)
