"""Case 2 - Logistic Regression (author: tu), chay theo CONTRIBUTING.md (ban moi: tuning tren xac suat) (cung quy trinh voi RF/XGBoost).

Tien xu ly = ban LogReg tot nhat truoc do (BAO_CAO_LOGREG Lan 3): 12 bien so -> signed log -> spline, ghep voi
signed log -> chuan hoa -> tuong tac bac 2; hod/dow one-hot. Chi tune C.

    python case2_logistic_regression_tu.py           # -> results/contrib/logistic_regression/
    python case2_logistic_regression_tu.py --quick   # chay thu (2 cau hinh, "test" = fold 0 cua train) -> results/contrib_quick/logistic_regression/

Quy trinh: train_folds.csv (143,771 dong, PredefinedSplit theo cot fold) -> RandomizedSearchCV(n_iter=30,
random_state=42, scoring = RMSE tren P(SCA=1), refit="rmse") -> cv_10fold tu chinh lan tuning
-> OOF cua cau hinh tot nhat -> p* (max macro-F1)
-> refit tren toan bo train -> du doan test.csv (70,814 dong) DUNG MOT LAN -> JSON.
"""
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             make_scorer, mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, PolynomialFeatures, SplineTransformer,
                                   StandardScaler)

_KAGGLE = sorted(Path("/kaggle/input").glob("**/train_folds.csv")) if Path("/kaggle/input").exists() else []
DATA = _KAGGLE[0].parent if _KAGGLE else Path(__file__).resolve().parent  # Kaggle: dataset ncamtu06/sca-folds
RESULTS = Path("/kaggle/working") if _KAGGLE else DATA / "results"
SEED = 42
QUICK = "--quick" in sys.argv
CASE2 = ["add_count", "remove_count", "view_count", "detail_count", "add_to_detail_ratio", "search_count",
         "unique_categories_viewed", "avg_price_bucket", "max_price_bucket", "unique_products_viewed",
         "avg_dwell_time", "session_duration", "hod", "dow"]
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)  # ung vien p*, chon tren OOF train
BASELINE_P = 0.7855
OUT = RESULTS / ("contrib_quick" if QUICK else "contrib") / "logistic_regression"
NAME = "logistic_regression_tu"

# ---------------- mo hinh ----------------
CAT = ["hod", "dow"]
NUM = [c for c in CASE2 if c not in CAT]


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def slog():
    return FunctionTransformer(signed_log, feature_names_out="one-to-one")


NUMERIC = make_union(
    make_pipeline(SimpleImputer(strategy="median"), slog(), SplineTransformer(n_knots=6, degree=3)),
    make_pipeline(SimpleImputer(strategy="median"), slog(), StandardScaler(),
                  PolynomialFeatures(2, interaction_only=True, include_bias=False), StandardScaler()))
PIPE = Pipeline([("prep", make_column_transformer((NUMERIC, NUM), (OneHotEncoder(handle_unknown="ignore"), CAT))),
                 ("clf", LogisticRegression(max_iter=3000, random_state=SEED))])
SPACE = {"clf__C": loguniform(1e-3, 1e3)}
SEARCH_JOBS = 4
PREPROCESSING = ("12 bien so: SimpleImputer(median) -> signed log -> SplineTransformer(n_knots=6, degree=3), ghep voi "
                 "SimpleImputer(median) -> signed log -> StandardScaler -> tuong tac bac 2 -> StandardScaler; "
                 "hod, dow: one-hot (handle_unknown=ignore). Tat ca trong Pipeline, chi tu 14 bien Case 2.")
NOTES = ("LogReg khong co trong bang phan cong CONTRIBUTING; ap dung dung quy trinh cua RF/XGBoost (RandomizedSearchCV "
         "n_iter=30, random_state=42, PredefinedSplit). Khong gian tham so: C ~ loguniform(1e-3, 1e3); tien xu ly co dinh "
         "= ban tot nhat truoc do (spline + tuong tac); solver lbfgs, max_iter 3000, class_weight=None. Tuning dung muc 5 CONTRIBUTING (RMSE_PROBA, refit theo RMSE). p* chon tren OOF train de "
         "toi da macro-F1; test chi du doan 1 lan sau refit. Chay tren Kaggle (CPU), phien ban sklearn ghi trong runtime.")


# ---------------- metric (CONTRIBUTING muc 6) ----------------
def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


# RMSE tren xac suat P(SCA=1) (CONTRIBUTING muc 5). sklearn < 1.4: thay response_method bang needs_proba=True
RMSE_PROBA = make_scorer(rmse, response_method="predict_proba", greater_is_better=False)


def reg_metrics(y, p):
    p = np.clip(p, 0, 1)
    mse = mean_squared_error(y, p)
    return {"rmse": float(np.sqrt(mse)), "mse": float(mse), "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p)), "roc_auc": float(roc_auc_score(y, p))}


def class_metrics(y, p, thr):
    pred = (np.clip(p, 0, 1) >= thr).astype(int)
    return {"threshold": float(thr),
            "precision_1": float(precision_score(y, pred, zero_division=0)),
            "recall_1": float(recall_score(y, pred, zero_division=0)),
            "f1_1": float(f1_score(y, pred, zero_division=0)),
            "precision_0": float(precision_score(y, pred, pos_label=0, zero_division=0)),
            "recall_0": float(recall_score(y, pred, pos_label=0, zero_division=0)),
            "f1_0": float(f1_score(y, pred, pos_label=0, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "pr_auc": float(average_precision_score(y, p)),
            "accuracy": float(accuracy_score(y, pred))}


def main():
    tr = pd.read_csv(DATA / "train_folds.csv")
    te = tr[tr.fold == 0] if QUICK else pd.read_csv(DATA / "test.csv")  # --quick khong dung den test.csv
    assert len(tr) == 143_771 and (QUICK or len(te) == 70_814) and sorted(tr.fold.unique()) == list(range(10))
    X, y, fold, Xte, yte = tr[CASE2], tr.SCA, tr.fold, te[CASE2], te.SCA
    cv = PredefinedSplit(fold)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"LogReg case2: train {X.shape} test {Xte.shape} | sklearn {sklearn.__version__}", flush=True)

    t0 = time.perf_counter()
    search = RandomizedSearchCV(PIPE, SPACE, n_iter=2 if QUICK else 30, random_state=SEED, cv=cv,
                                scoring={"rmse": RMSE_PROBA, "auc": "roc_auc"}, refit="rmse",
                                n_jobs=SEARCH_JOBS, verbose=2)
    search.fit(X, y)  # refit cau hinh tot nhat tren toan bo train
    tuning_s = time.perf_counter() - t0
    best = search.best_estimator_
    print(f"best {search.best_params_}  CV RMSE {-search.best_score_:.6f}  ({tuning_s / 60:.1f} min)", flush=True)

    t1 = time.perf_counter()
    oof = cross_val_predict(best, X, y, cv=cv, method="predict_proba", n_jobs=SEARCH_JOBS)[:, 1]
    oof_s = time.perf_counter() - t1
    i = search.best_index_  # cv_10fold = diem tung fold cua cau hinh tot nhat, lay tu chinh lan tuning (muc 6.3)
    pf = pd.DataFrame({"rmse": [-search.cv_results_[f"split{k}_test_rmse"][i] for k in range(10)],
                       "auc": [search.cv_results_[f"split{k}_test_auc"][i] for k in range(10)]})
    oof_rmse = [rmse(y[fold == k], oof[fold == k]) for k in range(10)]
    assert np.allclose(oof_rmse, pf.rmse, atol=1e-9), "OOF khong khop voi lan tuning"
    macro = np.array([f1_score(y, oof >= t, average="macro") for t in THRESHOLDS])
    p_star = float(THRESHOLDS[int(np.argmax(macro))])

    p_test = best.predict_proba(Xte)[:, 1]  # lan duy nhat dung test

    coef = pd.Series(best.named_steps["clf"].coef_[0], best.named_steps["prep"].get_feature_names_out())
    coef = coef.reindex(coef.abs().sort_values(ascending=False).index)
    res = {
        "case": "case2", "model": "logistic_regression", "author": "tu",
        "features": CASE2, "n_features": len(CASE2), "n_train": len(X), "n_test": len(Xte),
        "preprocessing": PREPROCESSING,
        "tuning": "RandomizedSearchCV(n_iter=30, random_state=42, cv=PredefinedSplit(train['fold']), "
                  "scoring={'rmse': RMSE_PROBA, 'auc': 'roc_auc'}, refit='rmse'), refit tren toan bo train",
        "param_distributions": {"clf__C": "loguniform(1e-3, 1e3)"},
        "best_params": {"solver": "lbfgs", "max_iter": 3000, "class_weight": None, "spline_n_knots": 6,
                        "spline_degree": 3, "n_model_inputs": int(coef.size),
                        **{k.split("__", 1)[1]: v for k, v in search.best_params_.items()}},
        "metrics_test": reg_metrics(yte, p_test),
        "metrics_class": {"p_star": p_star, "p_star_rule": "max macro-F1 tren OOF train, nguong 0.01-0.99 buoc 0.01",
                          "at_0.5": class_metrics(yte, p_test, 0.5), "at_p_star": class_metrics(yte, p_test, p_star)},
        "cv_10fold": {"rmse_mean": float(pf.rmse.mean()), "rmse_std": float(pf.rmse.std(ddof=0)),
                      "auc_mean": float(pf.auc.mean()), "auc_std": float(pf.auc.std(ddof=0)),
                      "rmse_folds": pf.rmse.tolist(), "auc_folds": pf.auc.tolist(), "std_ddof": 0,
                      "oof_macro_f1_at_p_star": float(macro.max()),
                      "oof_macro_f1_at_0.5": float(f1_score(y, oof >= 0.5, average="macro"))},
        "baseline_test": {"p": BASELINE_P, **reg_metrics(yte, np.full(len(yte), BASELINE_P))},
        "intercept": float(best.named_steps["clf"].intercept_[0]),
        "top15_coef": {k: float(v) for k, v in coef.head(15).items()},
        "converged": bool(best.named_steps["clf"].n_iter_[0] < 3000),
        "runtime": {"tuning_s": tuning_s, "oof_s": oof_s, "sklearn": sklearn.__version__,
                    "python": platform.python_version(), "machine": "kaggle" if _KAGGLE else "local"},
        "notebook": "case2_logistic_regression_tu.py",
        "notes": NOTES,
    }
    (OUT / f"{NAME}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=lambda v: v.item()))
    pd.DataFrame(search.cv_results_).drop(columns="params").sort_values("rank_test_rmse") \
        .to_csv(OUT / f"{NAME}_search.csv", index=False)
    pd.DataFrame({"fold": fold, "SCA": y, "p_oof": oof}).to_csv(OUT / f"{NAME}_oof.csv", index=False)
    pd.DataFrame({"SCA": yte, "p_test": p_test}).to_csv(OUT / f"{NAME}_test_pred.csv", index=False)
    joblib.dump(best, OUT / f"{NAME}.joblib", compress=3)
    print(json.dumps({k: res[k] for k in ("best_params", "cv_10fold", "metrics_test", "metrics_class")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
