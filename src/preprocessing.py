"""
src/preprocessing.py
Module làm sạch dữ liệu, xử lý giá trị khuyết (MAR), kỹ thuật đặc trưng và mã hóa dữ liệu.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH,
    TARGET_COLUMN, ID_COLUMN,
    BINARY_COLUMNS, NOMINAL_COLUMNS,
    REDUNDANT_VALUE_COLUMNS,
    RANDOM_STATE, TEST_SIZE
)


def load_raw_data(filepath=RAW_DATA_PATH) -> pd.DataFrame:
    """Đọc dữ liệu thô từ file CSV."""
    if not filepath.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {filepath}")
    df = pd.read_csv(filepath)
    return df


def handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý lỗi kiểu dữ liệu và giá trị khuyết ở cột TotalCharges.
    
    Phân loại Missing Data:
    - Loại: MAR (Missing At Random).
    - Lý do: 11 dòng bị khuyết TotalCharges đều có tenure = 0 (khách hàng mới đăng ký).
      Việc thiếu dữ liệu không ngẫu nhiên hoàn toàn (không phải MCAR), mà phụ thuộc 
      vào biến quan sát được là 'tenure'.
    - Giải pháp: Ép kiểu string -> numeric, sau đó điền khuyết bằng MonthlyCharges 
      (hoặc 0) thay vì xóa dòng để tránh mất mát thông tin nhóm khách hàng mới.
    """
    df = df.copy()

    # 1. Loại bỏ cột định danh không có giá trị dự đoán
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    # 2. Xử lý khoảng trắng ẩn trong TotalCharges -> biến thành NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 3. Imputation (Điền giá trị khuyết)
    # Vì tenure = 0 nên TotalCharges có thể xem là phí tháng đầu (MonthlyCharges)
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

    return df


def clean_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa các biến phân loại và đồng nhất nhãn dữ liệu.
    """
    df = df.copy()

    # 1. Chuyển SeniorCitizen từ 0/1 sang No/Yes để đồng bộ biến nhị phân
    if "SeniorCitizen" in df.columns and df["SeniorCitizen"].dtype != "object":
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # 2. Gộp các giá trị dư thừa 'No internet service' / 'No phone service' về 'No'
    for col in REDUNDANT_VALUE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].replace({
                "No internet service": "No",
                "No phone service": "No"
            })

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Engineering: Tạo thêm đặc trưng mới có ý nghĩa kinh doanh.
    - AvgChargePerMonth: Tổng phí chia cho thời gian sử dụng thực tế.
    - NumServices: Tổng số lượng dịch vụ khách hàng đang đăng ký sử dụng.
    """
    df = df.copy()

    # Phí trung bình mỗi tháng (cộng 1 vào tenure để tránh chia cho 0)
    df["AvgChargePerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Đếm số lượng dịch vụ phụ trợ
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    available_service_cols = [c for c in service_cols if c in df.columns]
    df["NumServices"] = df[available_service_cols].eq("Yes").sum(axis=1)

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mã hóa biến mục tiêu và các biến độc lập thành dạng số:
    - Target 'Churn': Yes -> 1, No -> 0
    - Binary features: Label Encoding (0/1)
    - Multi-category features: One-Hot Encoding (drop_first=True)
    """
    df = df.copy()

    # 1. Mã hóa Target: Yes -> 1, No -> 0
    if TARGET_COLUMN in df.columns:
        if df[TARGET_COLUMN].dtype != int and df[TARGET_COLUMN].dtype != np.int64 and df[TARGET_COLUMN].dtype != np.int32:
            df[TARGET_COLUMN] = (df[TARGET_COLUMN].astype(str) == "Yes").astype(int)

    # 2. Label Encoding cho biến nhị phân
    le = LabelEncoder()
    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # 3. One-Hot Encoding cho biến đa giá trị
    valid_nominal = [c for c in NOMINAL_COLUMNS if c in df.columns]
    df = pd.get_dummies(df, columns=valid_nominal, drop_first=True)

    # Chuyển các cột kiểu bool sinh ra từ get_dummies về kiểu int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý giá trị ngoại lệ bằng Capping (Winsorization) theo IQR, có log thống kê.

    Lưu ý (kết quả phân tích ở src/outliers.py và run_outliers.py):
    trên bộ Telco, 4 biến số liên tục KHÔNG có điểm nào vượt hàng rào 1.5*IQR, nên bước này
    giữ nguyên toàn bộ giá trị; nó đóng vai trò chốt an toàn nếu dữ liệu mới có ngoại lệ.
    Hàm này học ngưỡng trên toàn bộ df; khi cần tránh rò rỉ dữ liệu, dùng
    src.outliers.OutlierCapper (fit trên train, transform trên test).
    """
    from src.outliers import OUTLIER_COLUMNS, OutlierCapper

    capper = OutlierCapper(columns=OUTLIER_COLUMNS, method="iqr", k=1.5)
    df_out = capper.fit(df).transform(df)

    print("\n--- BÁO CÁO XỬ LÝ OUTLIERS (IQR METHOD) ---")
    for col, r in capper.report_.items():
        total = r["n_below"] + r["n_above"]
        if total > 0:
            print(f"[*] {col}: Capping {total} outliers "
                  f"({r['n_below']} dưới sàn, {r['n_above']} vượt trần). "
                  f"Khoảng chấp nhận: [{r['lower']:.2f}, {r['upper']:.2f}]")
        else:
            print(f"[*] {col}: Không phát hiện Outliers.")
    print("-------------------------------------------\n")
    return df_out


def preprocess_pipeline(df_raw: pd.DataFrame, save_processed=True) -> pd.DataFrame:
    """
    Quy trình tiền xử lý hoàn chỉnh từ Dữ liệu thô -> Dữ liệu sạch sẵn sàng huấn luyện.
    """
    df = handle_missing_data(df_raw)
    df = clean_categorical_values(df)
    df = create_features(df)
    df = handle_outliers(df)
    df = encode_features(df)

    if save_processed:
        df.to_csv(PROCESSED_DATA_PATH, index=False)

    return df


def split_and_scale_data(df_clean: pd.DataFrame):
    """
    Chia tập Train/Test theo tỷ lệ (stratified) và chuẩn hóa thang đo (StandardScaler).
    """
    X = df_clean.drop(columns=[TARGET_COLUMN])
    y = df_clean[TARGET_COLUMN]

    # Phân chia dữ liệu giữ nguyên tỷ lệ nhãn (stratify=y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Chuẩn hóa các biến số
    cols_to_scale = [
        col for col in ["tenure", "MonthlyCharges", "TotalCharges", "AvgChargePerMonth", "NumServices"]
        if col in X.columns
    ]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
