# 📊 Telco Customer Churn Prediction

> **Đồ án môn Khoa học Dữ liệu (Data Science Final Project)**  
> Phân tích hành vi và xây dựng mô hình dự báo khách hàng rời bỏ dịch vụ viễn thông.

---

## 📁 Cấu trúc Dự án Chuẩn (Standard Project Structure)

Dự án được tổ chức theo chuẩn cấu trúc thư mục Khoa học Dữ liệu chuyên nghiệp:

```text
project/
├── data/
│   ├── raw/                 # Dữ liệu gốc từ nguồn - KHÔNG BAO GIỜ chỉnh sửa trực tiếp
│   │   └── telco_churn.csv
│   ├── interim/             # Dữ liệu trung gian trong quá trình chuyển đổi (Work in progress)
│   └── processed/           # Dữ liệu đã làm sạch & mã hóa, sẵn sàng đưa vào mô hình
│       └── telco_cleaned.csv
├── notebooks/               # Sổ tay Jupyter đánh số theo thứ tự đọc/chạy
│   ├── 01_explore.ipynb     # Khám phá dữ liệu trực quan & phân phối biến (EDA)
│   ├── 02_clean.ipynb       # Phân tích giá trị khuyết (MAR) & Tiền xử lý dữ liệu
│   ├── 03_model.ipynb       # Huấn luyện, đánh giá & so sánh 4 mô hình Machine Learning
│   └── 04_outliers.ipynb    # Phân tích & xử lý Outlier (đơn biến, đa biến, thực nghiệm)
├── src/                     # Các hàm tái sử dụng dạng module chuẩn, không viết code spaghetti
│   ├── __init__.py
│   ├── config.py            # Đường dẫn thư mục, danh mục biến và siêu tham số
│   ├── preprocessing.py     # Quy trình làm sạch, xử lý Missing (MAR) và Encoding
│   ├── eda.py               # Các hàm trực quan hóa và xuất đồ thị phân tích
│   ├── modeling.py          # Huấn luyện mô hình, K-Fold CV, ROC curve & Feature Importance
│   └── outliers.py          # Phát hiện/xử lý Outlier: IQR, Z-score, Isolation Forest, LOF, OutlierCapper
├── reports/
│   ├── figures/             # Toàn bộ biểu đồ xuất bản phục vụ báo cáo (.png)
│   ├── outlier_analysis/    # Bảng số liệu phân tích Outlier (.csv, .json)
│   └── Bao_cao_Customer_Churn.md  # Báo cáo tổng kết đồ án chi tiết
├── requirements.txt         # Danh sách thư viện và phiên bản tương thích
├── main.py                  # Script thực thi toàn bộ pipeline chỉ với 1 câu lệnh
├── run_outliers.py          # Chạy riêng phần phân tích Outlier (python run_outliers.py)
└── README.md                # Tài liệu hướng dẫn tái lập dự án từ đầu (File này)
```

---

## 📌 Nguồn Dữ liệu (Data Sources)

* **Dataset:** Telco Customer Churn
* **Nguồn gốc:** [Kaggle - Telco Customer Churn by blastchar](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Kích thước:** 7,043 dòng $\times$ 21 cột thuộc tính (Nhân khẩu học, dịch vụ internet/thoại, hình thức thanh toán và trạng thái Churn).

---

## 🚀 Hướng dẫn Cài đặt & Tái lập Dự án (Reproduce from Scratch)

### 1. Yêu cầu Môi trường
* Python 3.10 trở lên (Đã kiểm thử và tương thích hoàn toàn với Python 3.13).

### 2. Cài đặt Thư viện
Mở Terminal/PowerShell tại thư mục gốc của dự án và chạy:

```bash
python -m pip install -r requirements.txt
```

### 3. Thực thi Toàn bộ Quy trình (End-to-End Pipeline)
Chỉ cần chạy file `main.py`, toàn bộ các bước:
1. Đọc dữ liệu từ `data/raw/`
2. Trực quan hóa EDA và lưu hình vào `reports/figures/`
3. Làm sạch dữ liệu, xử lý **MAR** ở `TotalCharges`, tạo đặc trưng mới và lưu vào `data/processed/`
4. Huấn luyện 4 thuật toán Machine Learning (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
5. Xuất bảng so sánh độ đo (ROC-AUC, F1-Score, Recall, Precision, 5-Fold CV) và biểu đồ ROC, Feature Importance.

```bash
python main.py
```

### 4. Sử dụng qua Jupyter Notebooks
Nếu muốn xem chi tiết từng bước có kèm giải thích học thuật và đồ thị tương tác:
* `notebooks/01_explore.ipynb` : Đọc và phân tích ban đầu.
* `notebooks/02_clean.ipynb` : Giải thích bản chất dữ liệu khuyết **MAR** (Missing At Random) và các bước tiền xử lý.
* `notebooks/03_model.ipynb` : Chi tiết huấn luyện mô hình và diễn giải kết quả kinh doanh.

---

## 🎯 Điểm Nhấn Kỹ thuật Cốt lõi (Key Highlights)

1. **Phân loại Dữ liệu Khuyết chuẩn học thuật:**  
   * Cột `TotalCharges` có 11 dòng bị rỗng do nhập khoảng trắng `" "`.
   * **Bản chất:** Là **MAR (Missing At Random)** vì 100% các dòng này có `tenure = 0` (khách mới đăng ký tháng đầu tiên, chưa phát sinh hóa đơn). Việc thiếu phụ thuộc vào biến quan sát được là `tenure`, không phải ngẫu nhiên hoàn toàn (MCAR) hay phụ thuộc vào chính giá trị ẩn (MNAR).
   * **Giải pháp:** Điền giá trị (`imputation`) bằng `MonthlyCharges` hoặc `0`, không drop dòng để bảo toàn thông tin khách hàng mới.
2. **Xử lý Mất cân bằng Lớp (Class Imbalance):**  
   * Tỷ lệ Churn: ~73.5% No vs ~26.5% Yes.
   * Áp dụng `class_weight='balanced'` trong thuật toán và ưu tiên tối ưu **ROC-AUC** cùng **Recall** thay vì Accuracy đơn thuần.
3. **Mã nguồn Chuẩn Module Hóa (Clean Architecture):**  
   * Tách bạch rõ rệt giữa cấu hình (`config.py`), tiền xử lý (`preprocessing.py`), đồ thị (`eda.py`), và mô hình (`modeling.py`).
3. **Phân tích Outlier 4 lớp:**  
   * Đơn biến (IQR, Z-score, MAD), đa biến (Isolation Forest, LOF, Mahalanobis), kiểm tra logic nghiệp vụ và thực nghiệm trên 3 mô hình.
   * **Kết luận:** không có outlier cần loại bỏ; Capping IQR giữ làm chốt an toàn, ngưỡng học trên tập train để tránh rò rỉ dữ liệu.
