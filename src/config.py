"""
src/config.py
Cấu hình đường dẫn, hằng số và nhóm biến cho toàn bộ dự án Customer Churn Prediction.
"""

from pathlib import Path

# ==========================================
# 1. ĐƯỜNG DẪN THƯ MỤC & FILE
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "telco_churn.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "telco_cleaned.csv"

REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Tự động tạo thư mục nếu chưa có
RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. HẰNG SỐ HUẤN LUYỆN (MODEL CONSTANTS)
# ==========================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ==========================================
# 3. ĐỊNH NGHĨA CÁC NHÓM BIẾN
# ==========================================
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

# Biến số liên tục
NUMERICAL_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Biến phân loại nhị phân (Yes/No hoặc Male/Female)
BINARY_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "PaperlessBilling"
]

# Biến phân loại đa giá trị (Cần One-Hot Encoding)
NOMINAL_COLUMNS = ["InternetService", "Contract", "PaymentMethod"]

# 7 cột dịch vụ có giá trị dư thừa ('No internet service' / 'No phone service')
REDUNDANT_VALUE_COLUMNS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]
