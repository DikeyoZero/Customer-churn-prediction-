import json
import os

def insert_cell_to_notebook(notebook_path, cell_type, source, position='append'):
    if not os.path.exists(notebook_path):
        print(f"Skipping {notebook_path}, file not found.")
        return
        
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    new_cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + '\n' for line in source.split('\n')]
    }
    
    if cell_type == "code":
        new_cell["outputs"] = []
        new_cell["execution_count"] = None
        
    # Remove trailing newlines from last line in source
    if new_cell["source"] and new_cell["source"][-1].endswith('\n'):
        new_cell["source"][-1] = new_cell["source"][-1][:-1]
        
    if position == 'append':
        nb['cells'].append(new_cell)
    else:
        nb['cells'].insert(position, new_cell)
        
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Updated {notebook_path}")

# 1. Update 01_explore.ipynb
explore_md = """## 5. Phân tích Giá trị ngoại lệ (Outliers) chuyên sâu
Việc xác định giá trị ngoại lệ giúp chúng ta hiểu rõ hơn về các biến động bất thường trong dữ liệu, như khách hàng VIP có cước phí cực cao hoặc những người dùng rất lâu năm. 
Chúng ta sẽ sử dụng phương pháp **Dải phân vị (IQR - Interquartile Range)** kết hợp Boxplot so sánh **Trước và Sau** khi xử lý (Capping)."""

explore_code = """from src.eda import plot_outliers_before_after
from IPython.display import Image, display

# Gọi hàm tính toán và vẽ biểu đồ Before/After
plot_outliers_before_after(df, save_path='../reports/figures/03b_outliers_before_after.png')

# Hiển thị kết quả trực quan
display(Image(filename='../reports/figures/03b_outliers_before_after.png'))"""

insert_cell_to_notebook("notebooks/01_explore.ipynb", "markdown", explore_md)
insert_cell_to_notebook("notebooks/01_explore.ipynb", "code", explore_code)

# 2. Update 02_clean.ipynb
clean_md = """## 4. Xử lý Giá trị Ngoại lệ (Outlier Treatment)
Trong lĩnh vực viễn thông, cước phí cao hoặc thâm niên (tenure) dài thường phản ánh khách hàng thực tế (VIP, khách hàng trung thành) thay vì lỗi đo lường. 
**Quyết định:** Sử dụng phương pháp **Winsorization (Capping)** dựa trên dải phân vị IQR thay vì xóa bỏ (Drop) để bảo toàn kích thước mẫu dữ liệu. Các giá trị nằm ngoài khoảng `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]` sẽ được kéo về bằng chính các biên này."""

clean_code = """from src.preprocessing import handle_outliers

# Xem phân phối trước khi xử lý
print("--- TRƯỚC KHI XỬ LÝ OUTLIERS ---")
display(df_cleaned[["tenure", "MonthlyCharges", "TotalCharges"]].describe())

# Áp dụng hàm Capping
df_cleaned = handle_outliers(df_cleaned)

# Xem phân phối sau khi xử lý
print("\\n--- SAU KHI CAPPING ---")
display(df_cleaned[["tenure", "MonthlyCharges", "TotalCharges"]].describe())"""

insert_cell_to_notebook("notebooks/02_clean.ipynb", "markdown", clean_md)
insert_cell_to_notebook("notebooks/02_clean.ipynb", "code", clean_code)
