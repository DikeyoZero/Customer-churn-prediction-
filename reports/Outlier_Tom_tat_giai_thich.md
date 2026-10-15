# TÓM TẮT & GIẢI THÍCH: PHẦN OUTLIER (GIÁ TRỊ NGOẠI LỆ)
**Đồ án:** Customer Churn Prediction (Telco) | **Phụ trách:** mảng Outlier

> Tài liệu này giúp bạn **hiểu để tự trình bày được**. Số liệu chi tiết nằm ở mục 3.3 của `Bao_cao_Customer_Churn.md` và thư mục `reports/outlier_analysis/`.

---

## PHẦN 0. TÓM TẮT 30 GIÂY

* **Outlier là gì?** Là giá trị nằm quá xa phần lớn dữ liệu, có thể do lỗi nhập liệu hoặc do khách hàng thật khác thường.
* **Dữ liệu Telco có outlier không?** **Không có outlier đơn biến.** Các điểm "lạ" ở góc nhìn đa biến chỉ là các nhóm khách hàng thật (VIP, khách mới, khách gói rẻ).
* **Đã làm gì?** Kiểm tra bằng 7 phương pháp, kiểm tra logic dữ liệu, rồi chạy thực nghiệm 8 cách xử lý trên 3 mô hình.
* **Quyết định:** **Giữ nguyên toàn bộ 7,043 khách hàng.** Capping IQR giữ lại như chốt an toàn (không đổi giá trị nào).
* **Phát hiện đáng nói:** Bước capping đã có trong project trước đó **không có tác dụng gì** (AUC chênh đúng 0.0000). Báo cáo cũ nói nó "giúp giảm nhiễu" là chưa chính xác; báo cáo mới đã sửa.

---

## PHẦN 1. KIẾN THỨC NỀN (để hiểu phần còn lại)

### 1.1. Outlier là gì, vì sao phải xử lý?
Ví dụ: 99 khách trả 20–120 USD/tháng và 1 khách trả 5,000 USD/tháng. Khách thứ 100 kéo trung bình lên cao và làm các mô hình nhạy với thang đo (Logistic Regression, SVM, KNN) bị lệch. Mô hình dạng cây (Random Forest, Gradient Boosting) ít bị ảnh hưởng hơn vì chỉ chia theo ngưỡng.

### 1.2. Hai loại nguyên nhân, hai cách xử lý khác nhau
| Nguyên nhân | Ví dụ | Nên làm |
|:---|:---|:---|
| **Lỗi dữ liệu** | tenure = −5, cước = 999,999 | Sửa hoặc xóa |
| **Giá trị thật nhưng hiếm** | khách VIP 72 tháng, cước 118 USD | **Giữ** (có thể cap nhẹ nếu cần) |

Đây là điểm mấu chốt: **không phải cứ outlier là xóa**. Phải hỏi "nó là lỗi hay là khách thật?" trước.

### 1.3. Các phương pháp phát hiện đã dùng
| Phương pháp | Cách hoạt động (nói đơn giản) | Ưu điểm | Nhược điểm |
|:---|:---|:---|:---|
| **IQR 1.5** | Q1 = mốc 25%, Q3 = mốc 75%, IQR = Q3 − Q1. Điểm nào ngoài [Q1 − 1.5·IQR, Q3 + 1.5·IQR] là outlier | Không cần dữ liệu phân phối chuẩn | Với dữ liệu lệch có thể báo thừa |
| **IQR 3.0** | Như trên nhưng hệ số 3 | Chỉ bắt outlier *rất* cực đoan | Bỏ sót outlier vừa |
| **Z-score** | Đo điểm cách trung bình bao nhiêu độ lệch chuẩn; \|z\| > 3 là outlier | Dễ hiểu | Chính outlier làm sai trung bình và độ lệch chuẩn |
| **Modified Z (MAD)** | Dùng trung vị thay trung bình | Bền vững hơn | Báo quá tay khi dữ liệu lệch (đã gặp ở `TotalCharges`) |
| **Isolation Forest** | Cắt ngẫu nhiên nhiều lần; điểm nào bị "cô lập" bằng ít nhát cắt là bất thường | Bắt được outlier đa biến | Phải chọn tỷ lệ contamination |
| **LOF** | So mật độ quanh một điểm với mật độ hàng xóm | Bắt outlier cục bộ | Nhạy với k |
| **Mahalanobis (MCD)** | Khoảng cách có tính đến tương quan giữa các biến | Bắt trường hợp "từng biến bình thường nhưng kết hợp lạ" | Giả định gần chuẩn |

### 1.4. Các cách xử lý
* **Xóa dòng (Drop):** đơn giản nhưng mất mẫu và mất thông tin.
* **Capping / Winsorization:** kéo giá trị vượt ngưỡng về ngưỡng (giữ nguyên số dòng).
* **Biến đổi (log):** nén đuôi dài của phân phối lệch.
* **Scaler bền vững (RobustScaler):** chuẩn hóa bằng trung vị và IQR nên ít bị outlier kéo.
* **Không làm gì:** hợp lệ nếu có bằng chứng outlier là dữ liệu thật.

---

## PHẦN 2. ĐÃ LÀM GÌ VÀ KẾT QUẢ

### Bước 1. Phát hiện đơn biến → **0 outlier**
Chạy 4 quy tắc cho `tenure`, `MonthlyCharges`, `TotalCharges`, `AvgChargePerMonth`.

* IQR 1.5, IQR 3.0, Z-score: **0 điểm** ở cả 4 biến.
* Modified Z báo 136 điểm ở `TotalCharges`, nhưng đây là **báo nhầm** do `TotalCharges` lệch phải (skewness 0.96). Đó là các khách lâu năm trả nhiều, không phải lỗi.
* **Vì sao không thể có?** Hàng rào IQR của `tenure` là [−60; 124] trong khi dữ liệu chỉ nằm trong 0–72. `tenure` bị chặn tự nhiên theo thời gian hoạt động, `MonthlyCharges` bị chặn bởi bảng giá dịch vụ.

> **Cách nói khi thuyết trình:** "Nhóm kiểm tra outlier bằng bốn quy tắc thống kê trên bốn biến số liên tục và không phát hiện điểm nào. Điều này hợp lý vì các biến bị giới hạn bởi bản chất nghiệp vụ."

### Bước 2. Phát hiện đa biến → **3 nhóm khách thật**
Một khách có thể bình thường ở từng biến nhưng lạ khi xét cùng lúc. Ba thuật toán cho ba nhóm khác nhau:

| Thuật toán | Nhóm bị gắn cờ | Churn | Ý nghĩa |
|:---|:---|:---:|:---|
| Isolation Forest (71 khách) | Tenure ≈ 70 tháng, cước ≈ 115 USD | **8.5%** | VIP lâu năm, trung thành |
| LOF (71 khách) | Chủ yếu tenure 0–3 tháng | **47.9%** | Khách mới, rủi ro rất cao |
| Mahalanobis (346 khách) | Tenure ≈ 66, cước ≈ 22 USD | **0.6%** | Khách lâu năm gói rẻ, gần như không rời |

Mức churn chung là 26.5%. Ba nhóm này có churn khác hẳn mức chung, nghĩa là **chứa thông tin rất hữu ích cho dự đoán**. Xóa chúng là xóa luôn tín hiệu.

### Bước 3. Kiểm tra logic → **không có lỗi nhập liệu**
Với khách có tenure > 0, `TotalCharges` phải xấp xỉ `tenure × MonthlyCharges`.

* Tỷ lệ thực tế nằm trong **0.69 – 1.57**, trung vị 1.00; không dòng nào ngoài [0.5; 2].
* 617 dòng lệch khỏi 1.0 là do khách **đổi gói cước giữa chừng**, hợp lý.
* Không trùng lặp, không giá trị âm, không tenure > 72.

### Bước 4. Thực nghiệm → **xử lý outlier gần như không đổi kết quả**
Thử 8 phương án trên 3 mô hình, lặp 10 lần chia train/test, ngưỡng chỉ học trên train (tránh rò rỉ dữ liệu).

| Phát hiện | Số liệu | Ý nghĩa |
|:---|:---|:---|
| Capping hiện tại = không xử lý | ΔAUC = **0.0000** | Bước capping không đổi giá trị nào |
| Mọi phương án gần như bằng nhau | Chênh lệch < 0.003 so với dao động ±0.013 | Không có ý nghĩa thực tiễn |
| Xóa outlier Isolation Forest làm hại | GB: −0.0024 | Xóa VIP mất tín hiệu |
| `log1p(TotalCharges)` có lợi nhẹ cho Logistic Regression | +0.0016 (p = 0.0025) | Xử lý *độ lệch*, không phải outlier |

---

## PHẦN 3. QUYẾT ĐỊNH VÀ LÝ DO

| Quyết định | Lý do |
|:---|:---|
| **Giữ toàn bộ 7,043 dòng** | Không có lỗi dữ liệu; các điểm lạ là khách thật; xóa không cải thiện mô hình |
| **Giữ Capping IQR 1.5 làm chốt an toàn** | Không đổi giá trị nào hiện tại, nhưng bảo vệ pipeline nếu có dữ liệu mới |
| **Dùng `OutlierCapper` (fit trên train) khi huấn luyện** | Tránh **rò rỉ dữ liệu** (data leakage): ngưỡng không được tính từ dữ liệu test |
| **Tùy chọn: `log1p(TotalCharges)` cho Logistic Regression** | Cải thiện nhẹ, nhưng đổi đặc trưng chung nên cần cả nhóm đồng ý |

### Rò rỉ dữ liệu là gì? (hay bị hỏi)
Nếu tính ngưỡng IQR trên **toàn bộ** dữ liệu rồi mới chia train/test, thông tin của tập test đã "lọt" vào bước tiền xử lý, làm điểm đánh giá lạc quan. Cách đúng: tính ngưỡng trên **train**, rồi áp cùng ngưỡng đó cho test. Trong project này hàm `handle_outliers` cũ tính trên toàn bộ dữ liệu, nhưng vì không cắt điểm nào nên kết quả không bị ảnh hưởng; module mới `OutlierCapper` làm đúng quy trình.

---

## PHẦN 4. CÂU HỎI THẦY/CÔ CÓ THỂ HỎI

1. **"Sao nhóm không xóa outlier?"** Vì không có outlier do lỗi. Các điểm lạ đa biến là khách thật và có tỷ lệ churn khác hẳn mức chung; thực nghiệm cho thấy xóa không cải thiện (Isolation Forest còn làm Gradient Boosting giảm AUC).
2. **"Sao Modified Z báo 136 điểm mà bạn vẫn nói không có outlier?"** Vì `TotalCharges` lệch phải; MAD nhỏ nên đuôi phải bị báo quá tay. Giá trị lớn nhất (8,685 USD) khớp với 72 tháng × khoảng 120 USD nên không phi lý.
3. **"Capping IQR để làm gì nếu không có outlier?"** Làm chốt an toàn cho dữ liệu mới. Đã đo: AUC chênh 0.0000 so với không capping.
4. **"Tại sao chọn IQR mà không phải Z-score?"** IQR không giả định phân phối chuẩn; dữ liệu `tenure` và `MonthlyCharges` có hai đỉnh nên không chuẩn.
5. **"Tại sao dùng nhiều thuật toán đa biến?"** Mỗi thuật toán nhìn một kiểu bất thường (toàn cục, mật độ cục bộ, tương quan) nên kết quả bổ sung nhau; thực tế chỉ 7 khách trùng từ hai thuật toán trở lên.
6. **"Kết quả có đáng tin không?"** Có 10 lần chia dữ liệu, test luôn nguyên vẹn, báo cáo cả độ lệch chuẩn. Hạn chế: p-value chưa hiệu chỉnh đa so sánh và 10 lần chia chồng lấn nên chỉ là gợi ý.

---

## PHẦN 5. GIỚI HẠN CẦN NÓI THẲNG
* Kết luận "không có outlier" áp dụng cho **bộ Telco này**; dữ liệu khác có thể khác.
* Thực nghiệm đo ảnh hưởng bằng ROC-AUC, Recall, F1 trên 3 mô hình (không gồm Decision Tree).
* Ngưỡng contamination 1% của Isolation Forest/LOF là lựa chọn của nhóm, không phải sự thật khách quan.
* `log1p(TotalCharges)` mới chỉ là đề xuất, chưa đưa vào pipeline chung.

---

## PHẦN 6. CÁCH CHẠY LẠI
```bash
pip install -r requirements.txt
python run_outliers.py                 # đầy đủ (~4 phút)
python run_outliers.py --skip-experiment   # chỉ phát hiện (vài giây)
```
Hoặc mở `notebooks/04_outliers.ipynb`.

**File liên quan:** `src/outliers.py`, `run_outliers.py`, `notebooks/04_outliers.ipynb`, `reports/outlier_analysis/*.csv`, `reports/figures/03b_outliers_detection.png`, `03c_multivariate_outliers.png`, `03d_outlier_treatment_comparison.png`.
