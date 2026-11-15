"""
CASE 1 - OLS baseline: Linear Probability Model, sai số chuẩn robust HC3 (statsmodels)
Người chạy: Quốc Kiệt

Chạy từ thư mục gốc repo:   python scripts/case1_ols_kiet.py
Cần: pandas numpy scikit-learn statsmodels

Theo CONTRIBUTING.md:
  - Dữ liệu: data/train_folds.csv (143.771 dòng, cột fold 0-9) / data/test.csv (70.814 dòng).
    Không chia lại, không xoá trùng, không xoá dòng. Test chỉ dùng 1 lần.
  - Thiếu giá trị: SimpleImputer(strategy="median") nằm trong Pipeline (fit trên phần train của từng fold).
  - CV: PredefinedSplit(train["fold"]); refit trên 9 fold, chấm fold còn lại; RMSE và AUC từng fold,
    báo cáo mean và np.std (ddof=0). Không gộp OOF để tính CV.
  - OLS không tuning. Dự đoán cắt về [0, 1] trước khi tính mọi metric.
  - metrics_test: rmse, mse, mae, r2, roc_auc trên P(SCA=1) của test (sklearn).
  - metrics_class: tại ngưỡng 0.5 và p*; p* chọn trên dự đoán OOF của train để tối đa macro-F1.
  - baseline_test: luôn dự đoán tỷ lệ SCA=1 của train (≈ 0.7855).
  - random_state = 42 (OLS không có thành phần ngẫu nhiên).

Đầu ra: results/case1/ols_kiet.json và results/case1/ols_kiet/*
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, roc_auc_score,
                             average_precision_score, precision_score, recall_score, f1_score,
                             balanced_accuracy_score, accuracy_score)
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train_folds.csv"
TEST_PATH = ROOT / "data" / "test.csv"
OUT_JSON = ROOT / "results" / "case1" / "ols_kiet.json"
OUT_DIR = ROOT / "results" / "case1" / "ols_kiet"
TARGET, SEED = "SCA", 42

FEATURES = ['add_count', 'remove_count', 'view_count', 'add_to_detail_ratio',
            'search_count', 'unique_categories_viewed', 'avg_price_bucket',
            'avg_dwell_time', 'session_duration', 'hod', 'dow']
NUMERIC = [c for c in FEATURES if c not in ("hod", "dow")]
P_STAR_GRID = np.round(np.arange(0.01, 1.0, 0.005), 3)


def hod_cyclic(X):
    h = np.asarray(X, dtype=float)
    return np.column_stack([np.sin(2 * np.pi * h[:, 0] / 24), np.cos(2 * np.pi * h[:, 0] / 24)])


class LPMHC3(BaseEstimator, RegressorMixin):
    """OLS (Linear Probability Model) bằng statsmodels, sai số chuẩn HC3. predict() trả về giá trị chưa cắt."""

    def fit(self, X, y):
        self.res_ = sm.OLS(np.asarray(y, dtype=float), sm.add_constant(np.asarray(X, dtype=float),
                                                                       has_constant="add")).fit(cov_type="HC3")
        return self

    def predict(self, X):
        return self.res_.predict(sm.add_constant(np.asarray(X, dtype=float), has_constant="add"))


def make_pipeline():
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("hod", FunctionTransformer(hod_cyclic, feature_names_out=lambda *_: ["hod_sin", "hod_cos"]), ["hod"]),
        ("dow", OneHotEncoder(categories=[[1, 2, 3, 4, 5, 6, 7]], drop="first",
                              sparse_output=False, handle_unknown="ignore"), ["dow"]),
    ], verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("ols", LPMHC3())])


def predict_proba1(model, X):
    return np.clip(model.predict(X), 0, 1)


def reg_metrics(y, p):
    mse = mean_squared_error(y, p)
    return {"rmse": float(np.sqrt(mse)), "mse": float(mse), "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p)), "roc_auc": float(roc_auc_score(y, p))}


def class_metrics(y, p, thr):
    yp = (p >= thr).astype(int)
    return {"threshold": float(thr),
            "precision_1": float(precision_score(y, yp, pos_label=1, zero_division=0)),
            "recall_1": float(recall_score(y, yp, pos_label=1, zero_division=0)),
            "f1_1": float(f1_score(y, yp, pos_label=1, zero_division=0)),
            "precision_0": float(precision_score(y, yp, pos_label=0, zero_division=0)),
            "recall_0": float(recall_score(y, yp, pos_label=0, zero_division=0)),
            "f1_0": float(f1_score(y, yp, pos_label=0, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, yp)),
            "macro_f1": float(f1_score(y, yp, average="macro", zero_division=0)),
            "pr_auc": float(average_precision_score(y, p)),
            "accuracy": float(accuracy_score(y, yp))}


def vif_table(Z):
    out = {}
    for c in Z.columns:
        others = sm.add_constant(Z.drop(columns=c).values, has_constant="add")
        r2 = sm.OLS(Z[c].values, others).fit().rsquared
        out[c] = 1 / (1 - r2)
    return pd.Series(out, name="VIF").sort_values(ascending=False)


if __name__ == "__main__":
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    Xtr, ytr = train[FEATURES], train[TARGET]
    Xte, yte = test[FEATURES], test[TARGET]
    print(f"Train {train.shape}, Test {test.shape}, SCA=1 (train) = {ytr.mean():.4f}")

    # 1) CV 10-fold theo PredefinedSplit + OOF cho p*
    ps = PredefinedSplit(train["fold"].values)
    oof = np.full(len(train), np.nan)
    rows = []
    for tr_idx, va_idx in ps.split():
        m = clone(make_pipeline()).fit(Xtr.iloc[tr_idx], ytr.iloc[tr_idx])
        p = predict_proba1(m, Xtr.iloc[va_idx])
        oof[va_idx] = p
        f = int(train["fold"].iloc[va_idx[0]])
        rows.append({"fold": f, **reg_metrics(ytr.iloc[va_idx], p)})
    cv = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    assert not np.isnan(oof).any()
    print("\nCV 10-fold:\n", cv.round(5).to_string(index=False))

    # 2) p* trên OOF train, tối đa macro-F1
    grid = pd.DataFrame([{"threshold": t,
                          "macro_f1": f1_score(ytr, (oof >= t).astype(int), average="macro", zero_division=0)}
                         for t in P_STAR_GRID])
    p_star = float(grid.loc[grid["macro_f1"].idxmax(), "threshold"])
    print(f"\np* = {p_star} (macro-F1 OOF = {grid['macro_f1'].max():.4f})")

    # 3) Fit trên toàn bộ train, đánh giá test 1 lần
    final = make_pipeline().fit(Xtr, ytr)
    p_test = predict_proba1(final, Xte)
    raw_test = final.predict(Xte)
    m_test = reg_metrics(yte, p_test)
    m_class = {"at_0.5": class_metrics(yte, p_test, 0.5), "at_p_star": class_metrics(yte, p_test, p_star)}
    base_p = float(ytr.mean())
    baseline = reg_metrics(yte, np.full(len(yte), base_p))
    print("\nTest:", {k: round(v, 5) for k, v in m_test.items()})
    print("Baseline:", {k: round(v, 5) for k, v in baseline.items()})

    # 4) Hệ số HC3, VIF
    names = list(final.named_steps["prep"].get_feature_names_out())
    res = final.named_steps["ols"].res_
    ci = np.asarray(res.conf_int())
    coef = pd.DataFrame({"coef": res.params, "se_HC3": res.bse, "z": res.tvalues, "p_value": res.pvalues,
                         "ci_low": ci[:, 0], "ci_high": ci[:, 1]}, index=["const"] + names)
    Z = pd.DataFrame(final.named_steps["prep"].transform(Xtr), columns=names)
    vif = vif_table(Z)
    out_of_range = float(((raw_test < 0) | (raw_test > 1)).mean())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coef.to_csv(OUT_DIR / "ols_coefficients.csv")
    vif.to_csv(OUT_DIR / "ols_vif.csv")
    cv.to_csv(OUT_DIR / "ols_cv10.csv", index=False)
    grid.to_csv(OUT_DIR / "ols_p_star_grid.csv", index=False)
    with open(OUT_DIR / "ols_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(f"{res.summary(xname=['const'] + names)}\n\n"
                 f"(R2 trong summary là R2 trên train, KHÔNG dùng để so sánh)\n\n"
                 f"VIF\n{vif.round(2).to_string()}\n\nCV 10-fold\n{cv.round(5).to_string(index=False)}\n\n"
                 f"p* = {p_star}\n\nTest\n{pd.Series(m_test).round(5).to_string()}\n\n"
                 f"Baseline (luôn dự đoán {base_p:.4f})\n{pd.Series(baseline).round(5).to_string()}\n")

    result = {
        "case": "case1",
        "model": "ols",
        "author": "kiet",
        "features": FEATURES,
        "n_features": len(FEATURES),
        "split": ("data/train_folds.csv (143,771) / data/test.csv (70,814), khong chia lai, khong xoa dong; "
                  "CV 10-fold theo cot fold (PredefinedSplit); seed 42"),
        "best_params": {},
        "metrics_test": m_test,
        "metrics_class": m_class,
        "cv_10fold": {"rmse_mean": float(cv["rmse"].mean()), "rmse_std": float(np.std(cv["rmse"], ddof=0)),
                      "auc_mean": float(cv["roc_auc"].mean()), "auc_std": float(np.std(cv["roc_auc"], ddof=0))},
        "baseline_test": baseline,
        "notebook": "notebooks/case1_ols_kiet.ipynb",
        "notes": (
            "Linear Probability Model (statsmodels OLS, cov_type='HC3'), khong tuning nen best_params rong. "
            "Bien doi: hod -> sin/cos(2*pi*hod/24); dow -> one-hot (bo dow=1 lam nhom goc). "
            "SimpleImputer(median) trong Pipeline cho cac bien so (chi avg_price_bucket co thieu). "
            "Du doan cat ve [0, 1] truoc khi tinh moi metric; "
            f"{out_of_range:.2%} du doan test nam ngoai [0, 1] truoc khi cat. "
            f"p* tim tren luoi 0.010-0.995 buoc 0.005, toi da macro-F1 tren OOF train (p*={p_star}). "
            f"Baseline = ti le SCA=1 cua train ({base_p:.6f}). "
            f"Train R2 (statsmodels) = {res.rsquared:.6f}, chi de tham khao. "
            "He so HC3, VIF, CV tung fold, luoi p* trong results/case1/ols_kiet/."
        ),
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nDa ghi {OUT_JSON.relative_to(ROOT)}")
