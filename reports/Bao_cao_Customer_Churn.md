# 📑 BÁO CÁO ĐỒ ÁN MÔN KHOA HỌC DỮ LIỆU
## ĐỀ TÀI: PHÂN TÍCH VÀ XÂY DỰNG MÔ HÌNH DỰ ĐOÁN KHÁCH HÀNG RỜI BỎ (CUSTOMER CHURN PREDICTION)

* **Học phần:** Khoa học Dữ liệu (Data Science)
* **Bộ dữ liệu:** Telco Customer Churn (Kaggle)
* **Kích thước dữ liệu:** 7,043 mẫu quan sát $\times$ 21 thuộc tính

---

## 1. TỔNG QUAN ĐỀ TÀI & BÀI TOÁN KINH DOANH

### 1.1. Bối cảnh
Trong ngành viễn thông (Telecom), chi phí để tìm kiếm một khách hàng mới (Customer Acquisition Cost - CAC) thường cao gấp 5 đến 7 lần so với chi phí giữ chân một khách hàng hiện hữu (Customer Retention Cost). Khách hàng hủy hợp đồng dịch vụ (Churn) gây thất thoát doanh thu nghiêm trọng. Do đó, việc xây dựng hệ thống phân tích dữ liệu và mô hình Machine Learning có khả năng nhận diện sớm những khách hàng có nguy cơ rời bỏ là bài toán then chốt giúp doanh nghiệp chủ động can thiệp bằng các chiến dịch chăm sóc khách hàng mục tiêu.

### 1.2. Mục tiêu đồ án
1. Khám phá và phân tích trực quan hành vi, dịch vụ và phương thức thanh toán của khách hàng (EDA).
2. Xử lý các vấn đề chất lượng dữ liệu: phát hiện lỗi kiểu dữ liệu, phân loại học thuật giá trị khuyết (**MAR**) và kỹ thuật đặc trưng (**Feature Engineering**).
3. Xây dựng và so sánh hiệu năng của 4 thuật toán phân loại: **Logistic Regression, Decision Tree, Random Forest, Gradient Boosting**.
4. Đánh giá mô hình dựa trên các độ đo phù hợp với dữ liệu mất cân bằng (**ROC-AUC, Recall, F1-Score**) và đề xuất chiến lược kinh doanh thực tiễn.

---

## 2. KHÁM PHÁ DỮ LIỆU (EXPLORATORY DATA ANALYSIS - EDA)

### 2.1. Phân phối biến mục tiêu (Target Imbalance)
* **Nhãn 0 (No Churn):** 5,174 khách hàng (~73.5%).
* **Nhãn 1 (Churn):** 1,869 khách hàng (~26.5%).
* **Nhận xét:** Tập dữ liệu có hiện tượng **mất cân bằng lớp (Class Imbalance)** với tỷ lệ xấp xỉ 3:1. Nếu sử dụng độ đo *Accuracy* (Độ chính xác) làm thước đo chính, một mô hình tầm thường luôn đoán "No" vẫn đạt 73.5% nhưng hoàn toàn vô dụng trong thực tế. Vì vậy, đồ án bắt buộc phải tập trung vào **ROC-AUC** và **Recall**.

### 2.2. Các nhân tố ảnh hưởng lớn nhất đến tỷ lệ Churn
1. **Loại Hợp đồng (Contract):**
   * Hợp đồng **Month-to-month (theo tháng):** Tỷ lệ rời bỏ lên tới **42.7%**.
   * Hợp đồng **One year (1 năm):** Giảm mạnh xuống còn **11.3%**.
   * Hợp đồng **Two year (2 năm):** Chỉ còn **2.8%**.
   * *Kết luận:* Ràng buộc hợp đồng dài hạn là yếu tố giữ chân mạnh mẽ nhất.

2. **Dịch vụ Internet (Internet Service):**
   * Khách hàng dùng **Fiber optic (Cáp quang):** Tỷ lệ rời bỏ đạt **41.9%** (cao gấp đôi mức trung bình toàn mạng).
   * Khách hàng dùng **DSL:** Tỷ lệ rời bỏ là **19.0%**.
   * Khách hàng **No Internet:** Tỷ lệ rời bỏ là **7.4%**.
   * *Insight:* Dịch vụ cáp quang có cước phí cao nhưng có thể chất lượng kết nối/dịch vụ chưa tương xứng, dẫn tới sự không hài lòng của khách hàng.

3. **Thời gian gắn bó (tenure):**
   * Nhóm khách hàng mới (tenure $\le$ 12 tháng) có tỷ lệ rời đi cao nhất (~47%). Càng sử dụng dịch vụ lâu năm, độ trung thành càng tăng rõ rệt.

4. **Dịch vụ hỗ trợ & Bảo mật (TechSupport, OnlineSecurity):**
   * Khách hàng **không đăng ký** bảo mật online hoặc hỗ trợ kỹ thuật có tỷ lệ rời bỏ cao gấp 3 lần nhóm có đăng ký.

---

## 3. LÀM SẠCH DỮ LIỆU & TIỀN XỬ LÝ (DATA PREPROCESSING)

### 3.1. Phân tích chuyên sâu Giá trị Khuyết (Missing Data Mechanism)

| Thuộc tính | Kiểu dữ liệu thô | Vấn đề phát hiện | Cơ chế Khuyết | Hướng giải quyết |
|:---|:---:|:---|:---:|:---|
| `customerID` | Object | Mã định danh duy nhất (unique 100%) | Không mang giá trị dự đoán | **Drop cột** để tránh overfitting |
| `TotalCharges` | Object (Chuỗi) | Chứa 11 ô có ký tự khoảng trắng `" "` | **MAR (Missing At Random)** | Ép kiểu số (`float`) + Điền khuyết bằng `MonthlyCharges` |
| `SeniorCitizen` | Int64 (0/1) | Biến phân loại nhưng lưu dạng số | Lỗi biểu diễn | Chuẩn hóa sang nhãn `"No"` / `"Yes"` |
| 7 cột dịch vụ | Object | Chứa nhãn phụ `"No internet/phone service"` | Dư thừa phân loại | Gộp về nhãn `"No"` đồng nhất |

#### 🎓 Luận giải học thuật về cơ chế MAR ở cột `TotalCharges`:
* **Khái niệm:** Dữ liệu khuyết được chia làm 3 cơ chế theo lý thuyết của Rubin (1976):
  * **MCAR (Missing Completely At Random):** Việc khuyết hoàn toàn ngẫu nhiên, không phụ thuộc vào bất kỳ biến nào.
  * **MAR (Missing At Random):** Việc khuyết phụ thuộc vào các biến quan sát được khác trong bộ dữ liệu, chứ không phụ thuộc vào giá trị chưa biết của chính biến đó.
  * **MNAR (Missing Not At Random):** Việc khuyết phụ thuộc trực tiếp vào chính giá trị ẩn bị khuyết (ví dụ: người thu nhập cao giấu thu nhập).
* **Bằng chứng thực nghiệm trong tập dữ liệu:**  
  Kiểm tra 11 dòng dữ liệu có `TotalCharges` bị khoảng trắng cho thấy: **100% các dòng này đều có `tenure = 0`** (khách hàng vừa mới đăng ký tháng đầu tiên, chưa đến kỳ xuất hóa đơn đầu tiên).
* **Kết luận:** Cơ chế khuyết của `TotalCharges` là **MAR** (phụ thuộc vào biến quan sát được là `tenure`).
* **Lý do KHÔNG dùng Listwise Deletion (Xóa dòng):** 11 dòng chỉ chiếm ~0.15% tổng dữ liệu, nhưng đây là nhóm khách hàng hoàn toàn mới. Nếu xóa bỏ sẽ làm mất thông tin nhóm mới gia nhập. Phương pháp tối ưu là **Imputation** (điền khuyết bằng `MonthlyCharges` hoặc `0` vì khách mới sử dụng dưới 1 tháng).

### 3.2. Kỹ thuật Đặc trưng (Feature Engineering)
* `AvgChargePerMonth`: $\text{TotalCharges} / (\text{tenure} + 1)$ nhằm phản ánh mức cước thực tế bình quân qua các tháng.
* `NumServices`: Tổng số lượng dịch vụ giá trị gia tăng mà khách hàng đăng ký (`PhoneService`, `Streaming`, `OnlineSecurity`,...).

### 3.3. Phân tích và Xử lý Giá trị ngoại lệ (Outliers)

> Mã nguồn: `src/outliers.py`, `run_outliers.py`, notebook `04_outliers.ipynb`. Kết quả số liệu: `reports/outlier_analysis/`. Hình: `03b`, `03c`, `03d` trong `reports/figures/`.

#### 3.3.1. Quy trình 4 lớp
| Lớp | Mục đích | Phương pháp |
|:---|:---|:---|
| 1. Đơn biến | Tìm giá trị cực đoan ở từng biến | IQR 1.5, IQR 3.0, Z-score, Modified Z-score (MAD) |
| 2. Đa biến | Tìm khách hàng bất thường khi xét nhiều biến cùng lúc | Isolation Forest, Local Outlier Factor, Mahalanobis robust (MCD) |
| 3. Logic nghiệp vụ | Phân biệt *lỗi dữ liệu* với *khách hàng thật* | Kiểm tra `TotalCharges` ≈ `tenure` × `MonthlyCharges`, trùng lặp, giá trị âm |
| 4. Thực nghiệm | Đo ảnh hưởng của từng cách xử lý tới mô hình | 8 phương án × 3 mô hình × 10 lần chia train/test |

#### 3.3.2. Kết quả phát hiện đơn biến (dữ liệu gốc, 7,043 khách hàng)
| Biến | Min – Max | Skewness | IQR 1.5 | IQR 3.0 | Z-score > 3 | Modified Z > 3.5 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `tenure` | 0 – 72 | 0.24 | 0 | 0 | 0 | 0 |
| `MonthlyCharges` | 18.25 – 118.75 | −0.22 | 0 | 0 | 0 | 0 |
| `TotalCharges` | 18.80 – 8,684.80 | 0.96 | 0 | 0 | 0 | 136 (*) |
| `AvgChargePerMonth` | 9.18 – 118.97 | −0.01 | 0 | 0 | 0 | 0 |

(*) Dương tính giả: `TotalCharges` lệch phải nên MAD nhỏ so với đuôi phải, khiến khách lâu năm bị đánh dấu oan; không có giá trị nào phi lý.

**Kết luận lớp 1:** Hàng rào IQR 1.5 của các biến nằm hoàn toàn ngoài miền giá trị thực (ví dụ `tenure`: [−60; 124] trong khi dữ liệu chỉ từ 0 đến 72), nên **không có outlier đơn biến**. Điều này khớp với bản chất dữ liệu: `tenure` bị chặn 0–72 tháng và `MonthlyCharges` bị chặn bởi bảng giá dịch vụ.

#### 3.3.3. Kết quả phát hiện đa biến
| Nhóm bị gắn cờ | Số khách | Tỷ lệ churn | Mô tả |
|:---|:---:|:---:|:---|
| Toàn bộ khách hàng | 7,043 | 26.5% | — |
| Isolation Forest (1%) | 71 | 8.5% | Khách VIP: tenure ≈ 70 tháng, cước ≈ $115 |
| LOF (1%) | 71 | 47.9% | Chủ yếu khách mới (tenure 0–3 tháng), mật độ thưa |
| Mahalanobis (MCD) | 346 | 0.6% | Khách lâu năm gói rẻ: tenure ≈ 66, cước ≈ $22 |

Ba thuật toán bắt ba nhóm khác nhau (chỉ 7 khách được ≥ 2 thuật toán cùng gắn cờ, không khách nào đủ cả 3). Mỗi nhóm đều là **phân khúc khách hàng thật**, có tỷ lệ churn khác biệt rõ rệt so với trung bình.

#### 3.3.4. Kiểm tra tính nhất quán
* Không có dòng trùng lặp, không có giá trị âm, không có `tenure` > 72.
* Tỷ lệ `TotalCharges / (tenure × MonthlyCharges)` nằm trong [0.69; 1.57], trung vị 1.00; **0 dòng** nằm ngoài [0.5; 2]. 617 dòng lệch khỏi 1.0 theo IQR là do khách đổi gói cước giữa chừng, không phải lỗi nhập liệu.

#### 3.3.5. Thực nghiệm ảnh hưởng tới mô hình
Thiết kế: 10 lần chia train/test (stratified 80/20, seed khác nhau); mọi ngưỡng và bộ lọc **chỉ học trên tập train** rồi áp lên test; tập test luôn nguyên vẹn. Kiểm định t ghép cặp so với baseline.

| Phương án | LR (AUC) | RF (AUC) | GB (AUC) |
|:---|:---:|:---:|:---:|
| A. Không xử lý (baseline) | 0.8459 | 0.8437 | 0.8419 |
| B. Capping IQR 1.5 (cách đang dùng) | 0.8459 | 0.8437 | 0.8419 |
| C. Winsorize 1% – 99% | 0.8459 | 0.8440 | 0.8422 |
| D. Winsorize 5% – 95% | 0.8454 | 0.8441 | 0.8425 |
| E. Loại outlier Isolation Forest (train) | 0.8459 | 0.8439 | 0.8395 |
| F. Loại outlier LOF (train) | 0.8459 | 0.8438 | 0.8421 |
| G. log1p(`TotalCharges`) | **0.8475** | 0.8437 | 0.8419 |
| H. RobustScaler thay StandardScaler | 0.8459 | 0.8437 | 0.8419 |

* **B trùng khớp tuyệt đối với A** (ΔAUC = 0.0000): xác nhận capping hiện tại không thay đổi giá trị nào.
* Mọi chênh lệch đều dưới 0.003, nhỏ hơn nhiều so với độ dao động giữa các lần chia (SD ≈ 0.013).
* Loại outlier bằng Isolation Forest làm giảm AUC của Gradient Boosting (−0.0024), cho thấy việc xóa nhóm VIP làm mất tín hiệu hữu ích.
* Chỉ `log1p(TotalCharges)` cải thiện Logistic Regression (+0.0016, p = 0.0025). Đây là xử lý *độ lệch phân phối*, không phải xử lý outlier, và mô hình dạng cây không bị ảnh hưởng.
* Lưu ý: p-value từ 10 lần chia chồng lấn là lạc quan và chưa hiệu chỉnh đa so sánh, chỉ nên coi là gợi ý.

#### 3.3.6. Quyết định xử lý
1. **Giữ toàn bộ 7,043 khách hàng**, không xóa dòng nào. Các điểm bị gắn cờ đa biến là khách VIP, khách mới và khách gói rẻ, đều mang thông tin về churn.
2. **Giữ Capping (Winsorization) theo IQR 1.5 như chốt an toàn** trong pipeline; trên dữ liệu hiện tại nó không đổi giá trị nào, nhưng bảo vệ mô hình nếu dữ liệu mới xuất hiện ngoại lệ.
3. **Chống rò rỉ dữ liệu:** khi tái sử dụng cho mô hình, dùng `OutlierCapper` (học ngưỡng trên train, áp lên test) thay vì tính ngưỡng trên toàn bộ dữ liệu.
4. Tùy chọn: dùng `log1p(TotalCharges)` cho Logistic Regression (cần cả nhóm đồng ý vì đổi đặc trưng chung).

### 3.4. Mã hóa & Chuẩn hóa (Encoding & Scaling)
* Biến nhị phân: **Label Encoding** (0/1).
* Biến đa danh mục (`InternetService`, `Contract`, `PaymentMethod`): **One-Hot Encoding** (`drop_first=True` để tránh đa cộng tuyến Dummy Variable Trap).
* Chuẩn hóa thang đo: **StandardScaler** ($\mu = 0, \sigma = 1$) áp dụng cho các biến số trước khi đưa vào Logistic Regression.

---

## 4. XÂY DỰNG VÀ ĐÁNH GIÁ MÔ HÌNH MACHINE LEARNING

### 4.1. Thiết kế thực nghiệm
* **Phân chia dữ liệu:** 80% Train (5,634 mẫu) và 20% Test (1,409 mẫu), cố định `random_state=42` và sử dụng `stratify=y` để bảo toàn tỷ lệ nhãn Churn.
* **Xử lý mất cân bằng lớp:** Áp dụng trọng số `class_weight='balanced'` trên hàm mất mát của các thuật toán.
* **Đánh giá tin cậy:** Sử dụng **Stratified 5-Fold Cross Validation** trên tập huấn luyện.

### 4.2. Bảng kết quả so sánh hiệu năng các mô hình

| Thuật toán | ROC-AUC | F1-Score | Recall (Nhãn 1) | Precision | Accuracy | 5-Fold CV AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **0.8477** | 0.6217 | 0.7888 | 0.5130 | 0.7452 | **0.8486** |
| **Decision Tree** | 0.8307 | 0.6251 | **0.8048** | 0.5110 | 0.7438 | 0.8268 |
| **Random Forest** | 0.8431 | **0.6278** | 0.7781 | 0.5262 | 0.7551 | 0.8450 |
| **Gradient Boosting** | 0.8446 | 0.5786 | 0.5214 | **0.6500** | **0.7984** | 0.8458 |

### 4.3. Phân tích kết quả học thuật
1. **Tại sao ưu tiên Recall hơn Precision?**
   * Trong bài toán Churn, hậu quả của **False Negative (Bỏ sót khách hàng sắp rời đi)** là công ty mất vĩnh viễn khách hàng và doanh thu tương lai.
   * Ngược lại, hậu quả của **False Positive (Dự đoán nhầm khách hàng trung thành là sắp rời đi)** chỉ là công ty gửi nhầm ưu đãi/khuyến mãi cho họ (thậm chí còn làm tăng sự hài lòng).
   * Do đó, **Recall cao là ưu tiên chiến lược**. Cả **Logistic Regression** (Recall ~78.9%) và **Random Forest** (Recall ~77.8%) đều đáp ứng rất xuất sắc tiêu chí này.
2. **Độ ổn định:** Điểm 5-Fold Cross Validation AUC của Logistic Regression (0.8486) và Random Forest (0.8450) rất sát với điểm trên tập Test, chứng minh mô hình **không bị Overfitting**.

### 4.4. Top 5 Đặc trưng quan trọng nhất (Feature Importance - Random Forest)
1. `TotalCharges` (Tổng chi phí tích lũy)
2. `tenure` (Thời gian khách hàng gắn bó)
3. `MonthlyCharges` (Mức cước hàng tháng)
4. `Contract_Two year` (Hợp đồng kỳ hạn 2 năm - tác động giữ chân cực mạnh)
5. `InternetService_Fiber optic` (Khách dùng cáp quang - tác động rời bỏ cao)

---

## 5. ĐỀ XUẤT GIẢI PHÁP KINH DOANH (ACTIONABLE RECOMMENDATIONS)

Dựa trên kết quả phân tích dữ liệu và mô hình Machine Learning, doanh nghiệp viễn thông nên triển khai 4 hành động chiến lược:

1. **Chiến lược chuyển đổi hợp đồng:**  
   Thiết kế chính sách khuyến mãi (giảm 10-15% cước hoặc tặng thêm dịch vụ giá trị gia tăng) cho các khách hàng đang ký hợp đồng theo tháng (Month-to-month) đồng ý chuyển đổi sang kỳ hạn 1 hoặc 2 năm.
2. **Nâng cao trải nghiệm dịch vụ Cáp quang (Fiber optic):**  
   Rà soát lại băng thông thực tế, độ trễ và dịch vụ hỗ trợ của gói Fiber optic. Nhóm này trả tiền nhiều nhất nhưng tỷ lệ rời bỏ cao nhất (~42%).
3. **Chăm sóc đặc biệt giai đoạn đầu (Onboarding 0 - 6 tháng):**  
   Thiết lập quy trình tự động gửi tin nhắn hỏi thăm, hướng dẫn sử dụng và hỗ trợ kỹ thuật tận nơi trong 6 tháng đầu tiên để vượt qua "vùng nguy cơ rời bỏ cao".
4. **Bán kèm các gói dịch vụ bảo vệ (Cross-selling):**  
   Cung cấp gói combo miễn phí thử nghiệm `TechSupport` và `OnlineSecurity`, vì dữ liệu chứng minh khách hàng có dịch vụ này có độ trung thành cao gấp 3 lần.

---

## 6. KẾT LUẬN

Đồ án đã hoàn thành toàn diện các mục tiêu đề ra:
* Chuẩn hóa cấu trúc thư mục dự án theo quy chuẩn Khoa học Dữ liệu.
* Làm rõ bản chất học thuật của cơ chế dữ liệu khuyết (**MAR**).
* Phân tích outlier theo 4 lớp (đơn biến, đa biến, logic nghiệp vụ, thực nghiệm): không có outlier cần loại bỏ, giữ nguyên toàn bộ 7,043 mẫu.
* Xây dựng pipeline tiền xử lý và huấn luyện tự động với 4 mô hình Machine Learning đạt độ phân biệt cao (**ROC-AUC ~ 0.85**).
* Cung cấp các khuyến nghị kinh doanh có cơ sở thực nghiệm vững chắc.
