# Quy ước chung (để so sánh công bằng)

Mọi mô hình phải chạy trên **cùng dữ liệu, cùng fold, cùng cách tuning và cùng cách tính metric**. Bất kỳ sai khác nào cũng phải ghi vào `notes` của file JSON.

## 1. Dữ liệu

| File | Số dòng | Cột | Vai trò |
|------|---------|-----|---------|
| `data/train_folds.csv` | 143.771 | 16 biến + `SCA` + `fold` (0–9) | Train |
| `data/test.csv` | 70.814 | 16 biến + `SCA` | Test, chỉ dùng **một lần** để đánh giá cuối |

- **Biến mục tiêu:** `SCA` nhị phân. SCA=1 là bỏ giỏ (78,55%), SCA=0 là mua (21,45%).
- **Không chia lại dữ liệu:** không dùng `train_test_split`. Bỏ cột `fold` khỏi X.
- **Không xoá dòng:** không `drop_duplicates()`, không `dropna()`. Mọi mô hình phải có đúng 143.771 dòng train và 70.814 dòng test.
  - Train có 63 dòng trùng, test có 22 dòng trùng, và 64 dòng test cũng có trong train. Nhóm giữ nguyên các dòng này.
- **Giá trị thiếu:** chỉ có ở hai cột giá.

  | Cột | Train | Test |
  |-----|-------|------|
  | `avg_price_bucket` | 147 | 70 |
  | `max_price_bucket` | 134 | 64 |

  - Điền bằng **median của train** bằng `SimpleImputer(strategy="median")`, đặt trong `Pipeline` để mỗi fold tự fit trên phần train của nó.
  - Test được điền bằng median của train, không tính median trên test.
  - **Mọi mô hình, kể cả XGBoost**, đều dùng dữ liệu đã điền.
- **Giá trị ngoài miền:** một số biến ở test lớn hơn giá trị lớn nhất của train (ví dụ `add_count` max 51 so với 28). Giữ nguyên, không cắt.

## 2. Bộ biến

- **Case 1 (11 biến):** `add_count, remove_count, view_count, add_to_detail_ratio, search_count, unique_categories_viewed, avg_price_bucket, avg_dwell_time, session_duration, hod, dow`
- **Case 2 (14 biến):** 16 biến gốc, bỏ `nevents` (bằng tổng 4 biến khác) và `unique_products_in_cart`:
  `add_count, remove_count, view_count, detail_count, add_to_detail_ratio, search_count, unique_categories_viewed, avg_price_bucket, max_price_bucket, unique_products_viewed, avg_dwell_time, session_duration, hod, dow`
- Được biến đổi biến theo từng mô hình (sin/cos cho `hod`, one-hot cho `dow`, spline…), nhưng chỉ từ các biến của case đó, và phải ghi lại cách biến đổi.
- Trong JSON, `features` ghi **tên biến gốc**, `n_features` là 11 hoặc 14.

## 3. Mô hình và phân công

| Case | Mô hình | Người | Cách dự đoán |
|------|---------|-------|--------------|
| 1 | OLS (Linear Probability Model, sai số chuẩn HC3) | Kiệt | Dự đoán tuyến tính, cắt về [0, 1] |
| 1 | XGBoost | Lâm | `XGBClassifier`, `predict_proba(X)[:, 1]` |
| 1 | Random Forest | Ngọc | `RandomForestClassifier`, `predict_proba(X)[:, 1]` |
| 2 | XGBoost | Trân | `XGBClassifier`, `predict_proba(X)[:, 1]` |
| 2 | Random Forest | Tú | `RandomForestClassifier`, `predict_proba(X)[:, 1]` |

## 4. Cross-validation

- **10-fold theo cột `fold`** có sẵn trong `data/train_folds.csv`. Các fold đã phân tầng theo SCA.
- Trong sklearn: `cv = PredefinedSplit(train["fold"])`.
- **Không** tự tạo `KFold` hay `StratifiedKFold`.

## 5. Tuning

- **OLS:** không tuning. Nếu có bước chọn biến thì chỉ làm trên train.
- **XGBoost và Random Forest** dùng cùng một cách:

  ```python
  import numpy as np
  from sklearn.metrics import make_scorer, mean_squared_error

  def rmse(y, p):
      return float(np.sqrt(mean_squared_error(y, p)))

  # RMSE trên xác suất P(SCA=1). sklearn < 1.4: thay response_method="predict_proba" bằng needs_proba=True
  RMSE_PROBA = make_scorer(rmse, response_method="predict_proba", greater_is_better=False)

  RandomizedSearchCV(model, param_distributions, n_iter=30, random_state=42,
                     cv=PredefinedSplit(train["fold"]),
                     scoring={"rmse": RMSE_PROBA, "auc": "roc_auc"}, refit="rmse")
  ```

- Tiêu chí chọn tham số là **RMSE trên xác suất P(SCA=1)**, trung bình 10 fold, tức cùng đại lượng với metric xếp hạng ở mục 6.1. AUC chỉ ghi lại để báo cáo ở `cv_10fold`.
- **Không** dùng `scoring="neg_root_mean_squared_error"`: với classifier, scorer này gọi `predict()` nên tính trên nhãn 0/1, tương đương tối đa accuracy (trái với mục 6.2).
- Không gian tham số được khác nhau giữa hai mô hình, nhưng phải ghi vào `notes`.
- Sau khi tune xong thì refit trên **toàn bộ** train, rồi dự đoán test **một lần duy nhất**.
- Early stopping (nếu dùng) chỉ lấy validation từ train, **không** dùng test.
- Ghi tham số tốt nhất vào `best_params`.

## 6. Metric

Mọi metric đều tính trên **xác suất dự đoán P(SCA=1)**, bằng `sklearn.metrics`: `metrics_test` và `metrics_class` trên tập test, `cv_10fold` trên fold validation của train. Tuning (mục 5) cũng chấm trên xác suất, không trên nhãn 0/1.

**6.1. Metric chính** (key `metrics_test`): `rmse`, `mse`, `mae`, `r2`, `roc_auc`
- Xếp hạng mô hình theo **RMSE** (tiêu chí chính, cũng là tiêu chí tuning). **ROC-AUC** là tiêu chí phụ, báo cáo kèm; nếu hai tiêu chí cho thứ hạng khác nhau thì lấy theo RMSE và nêu rõ sự khác biệt.
- MAE chỉ để tham khảo.
- **Không** lấy R² từ `statsmodels .summary()`, vì đó là R² trên train.

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score

p = np.clip(p_test, 0, 1)
mse = mean_squared_error(y_test, p)
metrics_test = {
    "rmse": float(np.sqrt(mse)),
    "mse": float(mse),
    "mae": float(mean_absolute_error(y_test, p)),
    "r2": float(r2_score(y_test, p)),
    "roc_auc": float(roc_auc_score(y_test, p)),
}
```

**6.2. Metric phân loại** (key `metrics_class`)
- Báo cáo ở hai ngưỡng: **0.5** và **p\***. p* được chọn trên dự đoán OOF của train để tối đa macro-F1, không chọn trên test.
- Các chỉ số: `precision_1, recall_1, f1_1, precision_0, recall_0, f1_0, balanced_accuracy, macro_f1, pr_auc, accuracy`
- Không dùng `accuracy` làm tiêu chí: luôn đoán 1 đã đạt 78,55%.
- `pr_auc` = `average_precision_score(y, p)` cho lớp SCA=1, tính trên xác suất nên giống nhau ở cả hai ngưỡng. Mức sàn của nó là 0,7855 (tỷ lệ SCA=1), không phải 0,5.
- Các metric còn lại phụ thuộc ngưỡng, nên không dùng để kết luận mô hình nào tốt hơn.

**6.3. Cross-validation** (key `cv_10fold`): `rmse_mean`, `rmse_std`, `auc_mean`, `auc_std`
- Là điểm của **cấu hình tốt nhất** trên 10 fold validation, lấy từ chính lần tuning ở mục 5 (RMSE và AUC trên xác suất). Với OLS: fit lại trên 9 fold, chấm trên fold còn lại.
- Tính RMSE/AUC **từng fold** rồi lấy mean và std (`np.std`, ddof=0), không gộp OOF lại rồi tính một lần.

**6.4. Baseline** (key `baseline_test`, cùng các chỉ số như `metrics_test`): mô hình luôn dự đoán 0,7855 (tỷ lệ SCA=1 của train). Baseline này cho RMSE ≈ 0,4105, R² = 0, ROC-AUC = 0,5.

**6.5. Số liệu:** ghi số thô, đủ chữ số thập phân, không làm tròn trong JSON.

## 7. Khác

- **Seed:** `random_state = 42` ở mọi nơi.
- **Tên file:** JSON đặt tên `<model>_<tên>.json` (ví dụ `xgboost_tran.json`). Notebook đặt tên `case<1|2>_<model>_<tên>`.
- **Code chạy lại được:** commit đủ mọi script mà notebook gọi tới.
- **Không commit dữ liệu gốc** trong `data/raw/`.
