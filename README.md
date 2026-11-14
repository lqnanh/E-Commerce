# So sánh OLS và ML (XGBoost, Random Forest) cho bài toán dự đoán bỏ giỏ hàng (SCA)

Repo dùng để các thành viên đăng kết quả chạy mô hình, sau đó gộp thành bảng so sánh chung.
Biến mục tiêu `SCA` là nhị phân; mọi mô hình được so sánh trên **xác suất dự đoán P(SCA=1)**. Quy ước chi tiết nằm trong `CONTRIBUTING.md`.

## Phân công
| Case | Mô hình | Người phụ trách |
|------|---------|-----------------|
| Case 1 (11 biến) | OLS | Kiệt |
| Case 1 (11 biến) | XGBoost | Lâm |
| Case 1 (11 biến) | Random Forest | Ngọc |
| Case 2 (14 biến) | XGBoost | Trân |
| Case 2 (14 biến) | Random Forest | Tú |

## Cấu trúc
```
data/train_folds.csv    tập train (143.771 dòng), có cột fold 0–9
data/test.csv           tập test (70.814 dòng), chỉ dùng một lần
data/raw/               dữ liệu gốc (KHÔNG commit, đã gitignore)
results/case1/          file kết quả JSON của Case 1
results/case2/          file kết quả JSON của Case 2
notebooks/              notebook của từng người (đặt tên: case1_xgboost_lam.ipynb)
shap_figures/           hình SHAP và feature importance
aggregate.py            gộp mọi JSON trong results/ thành bảng so sánh
results_template.json   mẫu file kết quả
```

## Cách đăng kết quả
1. Tạo nhánh riêng: `git checkout -b case1-xgboost-lam`
2. Chạy mô hình theo quy ước chung trong `CONTRIBUTING.md` (cùng dữ liệu, cùng fold, cùng cách tuning và metric).
3. Copy `results_template.json` vào `results/case1/` (hoặc `case2/`), đặt tên `<model>_<tên>.json`, điền số liệu.
4. Đưa notebook vào `notebooks/` và hình vào `shap_figures/`.
5. Commit, push, mở Pull Request. Một người khác trong nhóm xem qua rồi merge.

## Gộp kết quả
```
pip install pandas tabulate
python aggregate.py
```
Lệnh này tạo `summary.csv` và `summary.md` gồm RMSE, ROC-AUC, MSE, MAE, R² trên test và RMSE/AUC của CV 10-fold cho mọi mô hình, xếp theo RMSE, kèm mức cải thiện so với OLS.
