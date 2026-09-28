"""
main.py
Điểm thực thi chính (Entry Point) của đồ án Customer Churn Prediction.
Chạy toàn bộ quy trình: Tải dữ liệu -> Tiền xử lý (MAR) -> EDA -> Huấn luyện mô hình -> Xuất báo cáo.
"""

import sys
import time
import pandas as pd

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, FIGURES_DIR
from src.preprocessing import load_raw_data, preprocess_pipeline, split_and_scale_data
from src.eda import generate_all_eda_plots
from src.modeling import run_modeling_pipeline


def print_banner():
    banner = """
========================================================================
       ĐỒ ÁN KHOA HỌC DỮ LIỆU: CUSTOMER CHURN PREDICTION
       Pipeline: Data Cleaning (MAR) -> EDA -> Modeling & Evaluation
========================================================================
"""
    print(banner)


def main():
    start_time = time.time()
    print_banner()

    # ---------------------------------------------------------
    # BƯỚC 1: LOAD DỮ LIỆU THÔ
    # ---------------------------------------------------------
    print(f"[*] BƯỚC 1: Đọc dữ liệu từ: {RAW_DATA_PATH.relative_to(RAW_DATA_PATH.parent.parent)}")
    df_raw = load_raw_data()
    print(f"    -> Đã tải thành công: {df_raw.shape[0]:,} dòng, {df_raw.shape[1]} cột.\n")

    # ---------------------------------------------------------
    # BƯỚC 2: KHÁM PHÁ DỮ LIỆU & XUẤT BIỂU ĐỒ (EDA)
    # ---------------------------------------------------------
    print("[*] BƯỚC 2: Tiến hành EDA & Trực quan hóa dữ liệu...")
    generate_all_eda_plots(df_raw)
    print(f"    -> Biểu đồ khám phá đã được lưu tại: {FIGURES_DIR.relative_to(FIGURES_DIR.parent.parent)}\n")

    # ---------------------------------------------------------
    # BƯỚC 3: LÀM SẠCH & TIỀN XỬ LÝ (DATA CLEANING & FEATURE ENG)
    # ---------------------------------------------------------
    print("[*] BƯỚC 3: Làm sạch dữ liệu & Kỹ thuật đặc trưng (Feature Engineering)...")
    print("    -> Xử lý TotalCharges: Dạng Missing At Random (MAR), liên quan tới tenure = 0.")
    print("    -> Chuẩn hóa nhãn phân loại & Tạo thêm đặc trưng: AvgChargePerMonth, NumServices.")
    df_clean = preprocess_pipeline(df_raw, save_processed=True)
    print(f"    -> Dữ liệu sạch đã lưu tại: {PROCESSED_DATA_PATH.relative_to(PROCESSED_DATA_PATH.parent.parent)}")
    print(f"    -> Kích thước ma trận đặc trưng sau Encoding: {df_clean.shape}\n")

    # ---------------------------------------------------------
    # BƯỚC 4: PHÂN CHIA TRAIN/TEST & CHUẨN HÓA
    # ---------------------------------------------------------
    print("[*] BƯỚC 4: Phân chia tập Train/Test (Stratified 80/20) & Chuẩn hóa thang đo...")
    X_train, X_test, y_train, y_test, _ = split_and_scale_data(df_clean)
    print(f"    -> Tập Train: {X_train.shape[0]:,} mẫu | Tập Test: {X_test.shape[0]:,} mẫu\n")

    # ---------------------------------------------------------
    # BƯỚC 5: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH ML
    # ---------------------------------------------------------
    print("[*] BƯỚC 5: Huấn luyện và Đánh giá các Mô hình Machine Learning...")
    results = run_modeling_pipeline(X_train, X_test, y_train, y_test)

    # Hiển thị bảng tổng kết
    summary_data = []
    for model_name, res in results.items():
        summary_data.append({
            "Mô hình": model_name,
            "ROC-AUC": f"{res['ROC-AUC']:.4f}",
            "F1-Score": f"{res['F1-Score']:.4f}",
            "Recall": f"{res['Recall']:.4f}",
            "Precision": f"{res['Precision']:.4f}",
            "Accuracy": f"{res['Accuracy']:.4f}",
            "5-Fold CV AUC": f"{res['CV-ROC-AUC']:.4f}"
        })

    summary_df = pd.DataFrame(summary_data).set_index("Mô hình")
    print("\n" + "=" * 72)
    print("              BẢNG SO SÁNH KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 72)
    print(summary_df.to_string())
    print("=" * 72)

    elapsed_time = time.time() - start_time
    print(f"\n[+] HOÀN TẤT TOÀN BỘ QUY TRÌNH trong {elapsed_time:.2f} giây!")
    print(f"[+] Tất cả hình ảnh biểu đồ đã sẵn sàng trong thư mục: reports/figures/")


if __name__ == "__main__":
    main()
