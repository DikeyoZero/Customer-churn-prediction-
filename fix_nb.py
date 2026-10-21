import json

NB_PATH = r'C:\Users\HP\Data Science\notebooks\06_statistical_inference.ipynb'
with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── Giải thích Chi-square (chèn vào index 3, trước code cell) ────
explain_chi2 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 📖 Tại sao dùng Chi-square Test?\n",
        "\n",
        "| | |\n",
        "|---|---|\n",
        "| **Dùng khi nào** | Cả 2 biến đều là **phân loại** (categorical) |\n",
        "| **Biến kiểm tra** | `Contract` (Month-to-month / One year / Two year) |\n",
        "| **Biến mục tiêu** | `Churn` (Yes / No) |\n",
        "| **Câu hỏi** | Loại hợp đồng có **liên quan thật sự** đến Churn không, hay chỉ ngẫu nhiên? |\n",
        "\n",
        "**Cách hoạt động:**\n",
        "```\n",
        "Chi-square so sánh bảng thực tế với bảng kỳ vọng (nếu 2 biến độc lập):\n",
        "\n",
        "                  Churn=No    Churn=Yes\n",
        "Month-to-month     2220        1655     → 42.7% Churn ← RẤT CAO\n",
        "One year           1307         166     → 11.3% Churn\n",
        "Two year           1647          48     →  2.8% Churn ← RẤT THẤP\n",
        "\n",
        "Nếu H₀ đúng (không liên quan) → tỷ lệ 3 dòng phải gần bằng nhau.\n",
        "Thực tế chênh lệch rất lớn → chi² lớn → p-value nhỏ → Bác bỏ H₀.\n",
        "```\n",
        "> **Kết luận thực tế:** Khách dùng gói tháng có nguy cơ Churn **cao gấp 15 lần** gói 2 năm.\n"
    ]
}

# ── Giải thích T-test (chèn vào trước [06]) ──────────────────────
explain_ttest = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 📖 Tại sao dùng Independent T-test?\n",
        "\n",
        "| | |\n",
        "|---|---|\n",
        "| **Dùng khi nào** | So sánh **giá trị trung bình** của 1 biến số giữa **2 nhóm** |\n",
        "| **Biến số** | `MonthlyCharges` (phí hàng tháng — kiểu float) |\n",
        "| **2 nhóm** | Nhóm **Churn=Yes** vs nhóm **Churn=No** |\n",
        "| **Câu hỏi** | Nhóm Churn có trả phí **cao hơn thật sự** không, hay chỉ chênh lệch nhỏ do ngẫu nhiên? |\n",
        "\n",
        "**Cách hoạt động:**\n",
        "```\n",
        "         Trung bình Churn ($74.44) - Trung bình Not Churn ($61.27)\n",
        "t  =  ─────────────────────────────────────────────────────────────\n",
        "              Sai số chuẩn kết hợp của 2 nhóm\n",
        "\n",
        "t = 19.28 → chênh lệch lớn gấp 19 lần sai số → rất có ý nghĩa.\n",
        "```\n",
        "> **Kết luận thực tế:** Nhóm Churn trả trung bình **\\$74.44/tháng** — cao hơn nhóm ở lại **\\$13.17/tháng**.\n",
        "> Họ trả nhiều nhưng vẫn rời đi → **không hài lòng với chất lượng dịch vụ**.\n"
    ]
}

# ── Giải thích CI (chèn vào trước [09]) ──────────────────────────
explain_ci = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 📖 Tại sao dùng Confidence Interval?\n",
        "\n",
        "| | |\n",
        "|---|---|\n",
        "| **Dùng khi nào** | Ước tính **giá trị thật của tổng thể** từ mẫu |\n",
        "| **Vấn đề** | 7,043 khách hàng chỉ là **mẫu** — tỷ lệ Churn thật của toàn công ty là bao nhiêu? |\n",
        "| **Câu hỏi** | Với 95% chắc chắn, tỷ lệ Churn thật nằm trong khoảng nào? |\n",
        "\n",
        "**Công thức:**\n",
        "```\n",
        "p̂ = 26.5%   (tỷ lệ Churn đo được trong mẫu)\n",
        "n  = 7,043  (cỡ mẫu)\n",
        "z  = 1.96   (hệ số z cho mức tin cậy 95%)\n",
        "\n",
        "CI = p̂ ± z × √(p̂ × (1 - p̂) / n)\n",
        "   = 26.5% ± 1.03%\n",
        "   = [25.7% ; 27.5%]\n",
        "```\n",
        "> **Kết luận thực tế:** Ban lãnh đạo có thể tự tin báo cáo:\n",
        "> *\"Tỷ lệ khách rời bỏ của chúng ta là **26–27%**, không phải ước chừng.\"*\n",
        "> Khoảng hẹp (~2%) vì cỡ mẫu lớn → ước tính rất chính xác.\n"
    ]
}

# ── Giải thích A/B Test (chèn vào trước [11]) ────────────────────
explain_ab = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 📖 Tại sao dùng A/B Testing (Z-test 2 tỷ lệ)?\n",
        "\n",
        "| | |\n",
        "|---|---|\n",
        "| **Dùng khi nào** | So sánh **tỷ lệ %** giữa 2 nhóm (khác T-test — T-test so sánh trung bình) |\n",
        "| **Nhóm A** | Khách dùng **Fiber Optic** → tỷ lệ Churn là bao nhiêu %? |\n",
        "| **Nhóm B** | Khách dùng **DSL** → tỷ lệ Churn là bao nhiêu %? |\n",
        "| **Câu hỏi** | Fiber Optic có tỷ lệ Churn **cao hơn thật sự** không, hay chỉ là ngẫu nhiên? |\n",
        "\n",
        "**Phân biệt với T-test:**\n",
        "```\n",
        "T-test  → So sánh: Phí trung bình ($74 vs $61) — biến số liên tục\n",
        "Z-test  → So sánh: Tỷ lệ Churn (41.9% vs 19.0%) — biến tỷ lệ %\n",
        "```\n",
        "\n",
        "**Cách hoạt động:**\n",
        "```\n",
        "Nhóm A - Fiber Optic: 1,297 rời / 3,096 người = 41.9% Churn\n",
        "Nhóm B - DSL        :   459 rời / 2,421 người = 19.0% Churn\n",
        "Chênh lệch          :  41.9% - 19.0%          = +22.9%\n",
        "\n",
        "Z-test kiểm tra: chênh lệch 22.9% này có đủ lớn không\n",
        "so với biến động ngẫu nhiên của mẫu?\n",
        "→ z = 18.5, p < 0.001 → CÓ Ý NGHĨA.\n",
        "```\n",
        "> **Kết luận thực tế:** Đây là **bằng chứng thống kê** để yêu cầu bộ phận kỹ thuật\n",
        "> ưu tiên cải thiện dịch vụ Fiber Optic — không phải cảm tính mà có số liệu chứng minh.\n"
    ]
}

# ── Chèn vào đúng vị trí (chèn từ sau ra trước để index không lệch) ──
# Hiện tại: [02]=md chi2-title, [03]=code chi2, [04]=md output
#           [05]=md ttest-title, [06]=code ttest, [07]=md output
#           [08]=md ci-title,    [09]=code ci
#           [10]=md ab-title,    [11]=code ab,    [12]=md tổng kết

# Chèn ngược từ cuối lên đầu để index không bị lệch
nb['cells'].insert(11, explain_ab)   # trước code A/B test
nb['cells'].insert(9,  explain_ci)   # trước code CI
nb['cells'].insert(6,  explain_ttest) # trước code T-test
nb['cells'].insert(3,  explain_chi2) # trước code Chi-square

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Xong! Tong so o: {len(nb['cells'])}")
print("\nCau truc moi:")
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])[:70].replace('\n', ' ')
    print(f"  [{i:02d}] {c['cell_type']:8} | {src}")
