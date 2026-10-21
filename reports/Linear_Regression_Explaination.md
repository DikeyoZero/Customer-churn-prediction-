# TÓM TẮT & GIẢI THÍCH: PHẦN HỒI QUY TUYẾN TÍNH
**Đồ án:** Customer Churn Prediction (Telco) | Notebook: `notebooks/05_linear_regression.ipynb` | Mã: `src/regression.py`, `run_regression.py`

> Tài liệu giúp **hiểu để tự trình bày**. Số liệu chi tiết ở `reports/regression_analysis/` và các hình `13`–`16` trong `reports/figures/`.

---

## PHẦN 0. TÓM TẮT 30 GIÂY
* Hồi quy tuyến tính dự đoán **`MonthlyCharges`** (biến liên tục) rất chính xác: **test R² = 0,9988, RMSE = 1,05 USD** (baseline đoán trung bình: 30,09 USD).
* R² cao vì cước tháng gần như là **phép cộng bảng giá dịch vụ**, không phải vì mô hình "thông minh". Cần nói thẳng điều này.
* Bốn giả định: **tuyến tính, độc lập đạt**; **phương sai và độ chuẩn vi phạm về thống kê nhưng không đáng kể** (lệch ≈ 1 USD trên thang 30 USD).
* **Overfitting** được gây có chủ ý (150 mẫu, 252 đặc trưng bậc 2): train R² ≈ 1,0 nhưng RMSE test tăng 4,5 lần; CV báo trước đúng; Ridge/Lasso cứu.
* Hồi quy tuyến tính **không nên dùng cho `Churn`** (0/1): 16,6% dự đoán là xác suất âm → đồ án dùng Logistic Regression.

---

## PHẦN 1. BỐN MỤC TIÊU VÀ KẾT QUẢ

### Mục tiêu 1. Xây dựng và đọc mô hình
**Cách đọc hệ số:** giữ các biến khác cố định, tăng biến đó 1 đơn vị thì kỳ vọng `MonthlyCharges` đổi bằng hệ số. Biến 0/1 = chênh lệch giữa có và không có; biến one-hot = chênh lệch so với **nhóm gốc** (DSL, hợp đồng theo tháng, chuyển khoản tự động).

| Biến | Hệ số (USD/tháng) | Đọc |
|:--|--:|:--|
| Hằng số | 24,97 | Khách DSL, không điện thoại, không dịch vụ thêm |
| `PhoneService` | +20,03 | Có điện thoại |
| `MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport` | ≈ +5 mỗi cái | Mỗi dịch vụ thêm |
| `StreamingTV`, `StreamingMovies` | ≈ +10 mỗi cái | |
| `InternetService_Fiber optic` | +24,95 | Đắt hơn DSL |
| `InternetService_No` | −25,02 | Rẻ hơn DSL |
| `tenure`, nhân khẩu học, `Contract`, `PaymentMethod`, `PaperlessBilling` | ≈ 0 (p 0,43–0,99) | **Không ảnh hưởng cước tháng** (nhưng ảnh hưởng mạnh đến Churn) |

*Kiểm tra tay:* chỉ điện thoại = 24,97 + 20,03 − 25,02 = 19,98 USD, khớp giá thật (~21).
**Lưu ý:** p-value nhỏ chỉ nói hệ số khác 0 chắc chắn đến đâu, **độ lớn** đọc ở hệ số và khoảng tin cậy. Hệ số là tương quan có điều kiện, chưa phải nhân quả.

### Mục tiêu 2. Đánh giá trung thực
Nguyên tắc: **chỉ chấm trên dữ liệu chưa từng thấy**, luôn có **baseline**, dùng **CV** để chọn mô hình và chỉ chạm **test** một lần.

| Đánh giá | R² | RMSE (USD) | MAE (USD) |
|:--|--:|--:|--:|
| Baseline (đoán trung bình), test | −0,0003 | 30,09 | 26,03 |
| OLS, train | 0,9989 | 1,02 | 0,78 |
| OLS, **test** | **0,9988** | **1,05** | **0,79** |
| OLS, CV 5-fold | 0,9988 ± 0,0001 | 1,02 | – |

Train ≈ test ≈ CV → tổng quát hóa tốt. Đã loại `TotalCharges`, `AvgChargePerMonth` (tính từ biến mục tiêu → rò rỉ) và `NumServices` (đa cộng tuyến hoàn hảo).

### Mục tiêu 3. Chẩn đoán (LINE)
| Giả định | Kiểm tra | Kết quả | Kết luận thực tế |
|:--|:--|:--|:--|
| **L**inearity | RESET | p = 0,74 | Đạt |
| **I**ndependence | Durbin-Watson | 2,03 | Đạt (nhưng dữ liệu cắt ngang, DW ít ý nghĩa) |
| **N**ormality | Jarque-Bera / Shapiro | p ≈ 10⁻³⁸ / 10⁻¹¹; lệch 0,02; nhọn 3,86 | Vi phạm thống kê, gần chuẩn thực tế |
| **E**qual variance | Breusch-Pagan | p ≈ 10⁻¹⁰⁸; độ lệch chuẩn phần dư 0,67 → 1,23 USD | Vi phạm thống kê, nhỏ thực tế; báo cáo thêm sai số chuẩn vững HC3 |
| Đa cộng tuyến | VIF | lớn nhất 2,74 (thêm `NumServices`: ~10¹⁵) | Đạt |
| Điểm ảnh hưởng | Cook | lớn nhất 0,0042 | Không có dòng chi phối; khớp kết luận phần Outlier |

Mẫu lớn → kiểm định gần như luôn "vi phạm"; phải xem **biểu đồ** và **độ lớn**. Bốn giả định đảm bảo **suy luận** đáng tin, không phải để dự đoán tốt.

### Mục tiêu 4. Kiểm soát overfitting
| Mô hình (n_train = 150) | R² train | R² CV | R² test | RMSE test |
|:--|--:|--:|--:|--:|
| OLS bậc 1 | 0,9989 | 0,9984 | 0,9986 | 1,11 |
| OLS bậc 2 (252 đặc trưng) | 1,0000 | 0,9770 | 0,9719 | **5,04** |
| Ridge bậc 2 | 0,9999 | 0,9883 | 0,9921 | 2,68 |
| Lasso bậc 2 | 0,9991 | 0,9977 | 0,9984 | 1,20 |

* Train là thước đo vô dụng; **CV báo trước đúng** (0,977 sát test 0,972).
* **Ridge** co hệ số, **Lasso** ép về 0. Trên toàn bộ train, điều chuẩn **không đổi điểm** (RMSE 1,0513 / 1,0513 / 1,0510) nhưng Lasso ép về 0 đúng **11 biến** mà OLS báo không có ý nghĩa.
* Liên hệ đồ án: `LogisticRegression(C=1.0)` là L2; `max_depth`, `min_samples_leaf` là điều chuẩn của cây.

---

## PHẦN 2. KHI NÀO KHÔNG DÙNG: VÍ DỤ CHURN
OLS dự đoán Churn: **234/1.409 khách (16,6%) có "xác suất" âm** (thấp nhất −0,235); R² test 0,26; phần dư chia hai đường song song; Breusch-Pagan p ≈ 10⁻²²⁵. Về **xếp hạng** OLS gần bằng Logistic (AUC 0,834 vs 0,839), nhưng **ý nghĩa con số** sai → dùng Logistic Regression.

---

## PHẦN 3. CÂU HỎI THẦY/CÔ CÓ THỂ HỎI
1. **"R² 0,9988 có đáng tin?"** Có (tính trên test và CV, không rò rỉ), nhưng cao vì cước là bảng giá cộng dồn.
2. **"Breusch-Pagan p rất nhỏ, mô hình sai?"** Vi phạm thống kê nhưng nhỏ về thực tế; mẫu lớn làm kiểm định quá nhạy; đã đề xuất HC3.
3. **"Ridge khác Lasso?"** Ridge co về gần 0, Lasso ép về đúng 0 (chọn biến).
4. **"Sao không dùng hồi quy tuyến tính cho Churn?"** Xác suất âm, phần dư hai đường, phương sai $p(1-p)$ thay đổi.
5. **"Liên hệ phần Outlier?"** Cook lớn nhất 0,0042 → không có điểm chi phối, nhất quán với việc giữ 7.043 khách; ngưỡng capping học trên train theo cùng nguyên tắc đánh giá trung thực.

## PHẦN 4. GIỚI HẠN CẦN NÓI THẲNG
* R² ≈ 0,999 do bản chất dữ liệu, không phải bằng chứng mô hình mạnh ở bài toán khó.
* Overfitting ở mục 4 là **cố ý gây ra** để minh họa, không phải lỗi pipeline.
* Chỉ một lần chia (`random_state=42`) và một mẫu con 150 dòng; chưa lặp nhiều seed.
* Phần Churn dùng chia có phân tầng và Logistic bản đơn giản (không `class_weight`), AUC có thể lệch nhẹ so với báo cáo chính.

## PHẦN 5. CÁCH CHẠY LẠI
```bash
pip install -r requirements.txt     # đã thêm statsmodels
python run_regression.py            # ~6 giây
```
Hoặc mở `notebooks/05_linear_regression.ipynb`.
