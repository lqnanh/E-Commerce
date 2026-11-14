"""
CASE 1 - OLS baseline (Linear Probability Model, SE robust HC3)
Chạy từ thư mục gốc repo:  python notebooks/case1_ols_kiet.py
Cần: pandas numpy scipy scikit-learn

Dữ liệu (KHÔNG commit, đặt trong data/raw/):
  - data/raw/train_folds.csv : train.csv của nhóm + cột `fold` (0-9) dùng chung cho CV 10-fold
  - data/raw/test.csv        : tập test của nhóm, chỉ dùng 1 lần để tính metric test

Quy ước (README + CONTRIBUTING):
  - Giữ nguyên toàn bộ dòng (không drop_duplicates/dropna) để khớp fold ID và tập test chung.
  - Thiếu avg_price_bucket -> điền median, median chỉ fit trên phần train (từng fold / toàn train).
  - hod -> sin/cos, dow -> dummy (bỏ dow=1 làm nhóm gốc).
  - Metric: RMSE, MSE, MAE, R² bằng sklearn trên tập test; dự báo cắt về [0, 1].

Đầu ra: results/case1/ols_kiet.json và results/case1/ols_kiet/*
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, roc_auc_score,
                             accuracy_score, precision_score, recall_score, f1_score)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train_folds.csv"
TEST_PATH = ROOT / "data" / "raw" / "test.csv"
OUT_DIR = ROOT / "results" / "case1" / "ols_kiet"
TARGET, SEED = "SCA", 42
FEATURES = ['add_count', 'remove_count', 'view_count', 'add_to_detail_ratio',
            'search_count', 'unique_categories_viewed', 'avg_price_bucket',
            'avg_dwell_time', 'session_duration', 'hod', 'dow']
DOW_LEVELS = [1, 2, 3, 4, 5, 6, 7]


def prep(d, medians):
    """Điền median (fit trên train), hod -> sin/cos, dow -> dummy cố định cột."""
    d = d[FEATURES].fillna(medians).copy()
    d["hod_sin"] = np.sin(2 * np.pi * d["hod"] / 24)
    d["hod_cos"] = np.cos(2 * np.pi * d["hod"] / 24)
    d["dow"] = pd.Categorical(d["dow"], categories=DOW_LEVELS)
    d = d.drop(columns="hod")
    return pd.get_dummies(d, columns=["dow"], drop_first=True, dtype=float)


def design(X):
    return np.column_stack([np.ones(len(X)), X.values])


def ols_hc3(X, y):
    """OLS + sai số chuẩn robust HC3 (giống statsmodels cov_type='HC3')."""
    Xc = design(X)
    XtX_inv = np.linalg.inv(Xc.T @ Xc)
    beta = XtX_inv @ Xc.T @ y.values
    resid = y.values - Xc @ beta
    h = np.einsum("ij,jk,ik->i", Xc, XtX_inv, Xc)
    u = Xc * (resid / (1 - h))[:, None]
    se = np.sqrt(np.diag(XtX_inv @ (u.T @ u) @ XtX_inv))
    z = beta / se
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    n, k = Xc.shape
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    coef = pd.DataFrame({"coef": beta, "se_HC3": se, "z": z, "p_value": p,
                         "ci_low": beta - 1.96 * se, "ci_high": beta + 1.96 * se},
                        index=["const"] + list(X.columns))
    return coef, r2, 1 - (1 - r2) * (n - 1) / (n - k), beta


def predict(X, beta):
    return np.clip(design(X) @ beta, 0, 1)


def vif_table(X):
    res = {}
    for c in X.columns:
        others = design(X.drop(columns=c))
        b = np.linalg.lstsq(others, X[c].values, rcond=None)[0]
        r = X[c].values - others @ b
        res[c] = 1 / ((r ** 2).sum() / ((X[c] - X[c].mean()) ** 2).sum())
    return pd.Series(res, name="VIF").sort_values(ascending=False)


def reg_metrics(y, p):
    mse = mean_squared_error(y, p)
    return {"rmse": float(np.sqrt(mse)), "mse": float(mse),
            "mae": float(mean_absolute_error(y, p)), "r2": float(r2_score(y, p))}


def class_metrics(y, p):
    yp = (p >= 0.5).astype(int)
    return {"roc_auc": float(roc_auc_score(y, p)), "accuracy": float(accuracy_score(y, yp)),
            "precision": float(precision_score(y, yp, zero_division=0)),
            "recall": float(recall_score(y, yp)), "f1": float(f1_score(y, yp))}


if __name__ == "__main__":
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"Train {train.shape}, Test {test.shape}")

    # 1) CV 10-fold theo fold dùng chung
    cv = []
    for f in sorted(train["fold"].unique()):
        tr, va = train[train["fold"] != f], train[train["fold"] == f]
        med = tr[FEATURES].median()
        _, _, _, b = ols_hc3(prep(tr, med), tr[TARGET])
        cv.append({"fold": int(f), **reg_metrics(va[TARGET], predict(prep(va, med), b))})
    cv = pd.DataFrame(cv)
    print("\nCV 10-fold:\n", cv.round(5).to_string(index=False))

    # 2) Fit trên toàn bộ train, đánh giá 1 lần trên test
    med = train[FEATURES].median()
    A, B = prep(train, med), prep(test, med)
    coef, r2_train, r2adj, beta = ols_hc3(A, train[TARGET])
    p_te = predict(B, beta)
    m_test = reg_metrics(test[TARGET], p_te)
    c_test = class_metrics(test[TARGET], p_te)
    vif = vif_table(A)
    print(f"\nTrain R2={r2_train:.4f}  AdjR2={r2adj:.4f}")
    print("Test:", {k: round(v, 5) for k, v in {**m_test, **c_test}.items()})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coef.to_csv(OUT_DIR / "ols_coefficients.csv")
    vif.to_csv(OUT_DIR / "ols_vif.csv")
    cv.to_csv(OUT_DIR / "ols_cv10.csv", index=False)
    pd.DataFrame({"Test": {**m_test, **c_test}}).T.to_csv(OUT_DIR / "ols_metrics.csv")
    with open(OUT_DIR / "ols_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Train R2={r2_train:.4f} AdjR2={r2adj:.4f} (n={len(A)})\n\n"
                 f"{coef.round(4).to_string()}\n\nVIF\n{vif.round(2).to_string()}\n\n"
                 f"CV 10-fold\n{cv.round(5).to_string(index=False)}\n\n"
                 f"Test\n{pd.Series({**m_test, **c_test}).round(5).to_string()}\n")

    result = {
        "case": "case1",
        "model": "ols",
        "author": "kiet",
        "features": FEATURES,
        "n_features": len(FEATURES),
        "split": (f"Provided train.csv ({len(train):,} rows) / test.csv ({len(test):,} rows); seed=42. "
                  "Ten-fold CV on train using the shared fold assignments (train_folds.csv). "
                  "Test used only once."),
        "best_params": {"type": "Linear Probability Model", "se": "HC3 robust",
                        "pred_clip": [0, 1], "impute": "median (fit on train)",
                        "hod": "sin/cos", "dow": "dummy, base=1"},
        "metrics_test": m_test,
        "cv_10fold": {"rmse_mean": float(cv["rmse"].mean()), "rmse_std": float(cv["rmse"].std(ddof=0))},
        "notebook": "notebooks/case1_ols_kiet.py",
        "notes": (f"Train R2={r2_train:.6f}, Adj R2={r2adj:.6f}. Mean CV MSE={cv['mse'].mean():.6f}; "
                  f"mean CV MAE={cv['mae'].mean():.6f}. Test ROC-AUC={c_test['roc_auc']:.6f}, "
                  f"accuracy={c_test['accuracy']:.6f}, precision={c_test['precision']:.6f}, "
                  f"recall={c_test['recall']:.6f}, f1={c_test['f1']:.6f} (cutoff 0.50). "
                  "All rows kept (no dedup/dropna) to match shared folds and test set; "
                  "missing avg_price_bucket imputed with train median. "
                  "Coefficients, VIF, per-fold CV in results/case1/ols_kiet/."),
    }
    (ROOT / "results" / "case1" / "ols_kiet.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nĐã ghi results/case1/ols_kiet.json")
