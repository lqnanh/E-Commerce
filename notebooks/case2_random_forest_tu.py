"""Case 2 - Random Forest (author: tu), chay theo CONTRIBUTING.md (ban moi: tuning tren xac suat).

    python case2_random_forest_tu.py           # -> results/contrib/random_forest/
    python case2_random_forest_tu.py --quick   # chay thu (2 cau hinh, 20 cay, "test" = fold 0 cua train) -> results/contrib_quick/random_forest/

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             make_scorer, mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline

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
OUT = RESULTS / ("contrib_quick" if QUICK else "contrib") / "random_forest"
NAME = "random_forest_tu"

# ---------------- mo hinh ----------------
N_TREES = 20 if QUICK else 200
PIPE = Pipeline([("imp", SimpleImputer(strategy="median")),  # median cua phan train trong tung fold
                 ("clf", RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1, random_state=SEED))])
SPACE = {"clf__max_features": ["sqrt", 0.3, 0.5, 0.7, 0.9],
         "clf__min_samples_leaf": [5, 10, 15, 20, 25, 35, 50, 75, 100]}
SEARCH_JOBS = 1  # cay da chay song song ben trong moi lan fit (may it RAM)
NOTES = ("Khong gian tham so RF: max_features {sqrt,0.3,0.5,0.7,0.9} x min_samples_leaf {5,10,15,20,25,35,50,75,100} "
         "(45 to hop, lay ngau nhien 30); n_estimators co dinh 200; class_weight=None. Bo min_samples_leaf 1-3 vi "
         "lan truoc kem ro va cay qua lon so voi RAM may. Tuning dung muc 5 CONTRIBUTING (RMSE_PROBA, refit theo RMSE). "
         "p* chon tren OOF train (cung 10 fold) de toi da macro-F1; test chi du doan 1 lan sau refit. "
         "Chay tren Kaggle (CPU), phien ban sklearn ghi trong runtime. Thay ban cu (GridSearchCV 9 cau hinh, scoring roc_auc, Kaggle sklearn 1.6.1).")


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
    print(f"RF case2: train {X.shape} test {Xte.shape} | sklearn {sklearn.__version__}", flush=True)

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

    imp = pd.Series(best.named_steps["clf"].feature_importances_, CASE2).sort_values(ascending=False)
    res = {
        "case": "case2", "model": "random_forest", "author": "tu",
        "features": CASE2, "n_features": len(CASE2), "n_train": len(X), "n_test": len(Xte),
        "preprocessing": "SimpleImputer(median) trong Pipeline (fit lai moi fold); khong bien doi bien khac",
        "tuning": "RandomizedSearchCV(n_iter=30, random_state=42, cv=PredefinedSplit(train['fold']), "
                  "scoring={'rmse': RMSE_PROBA, 'auc': 'roc_auc'}, refit='rmse'), refit tren toan bo train",
        "param_distributions": SPACE,
        "best_params": {"n_estimators": N_TREES, "class_weight": None, "random_state": SEED,
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
        "feature_importance": {k: float(v) for k, v in imp.items()},
        "runtime": {"tuning_s": tuning_s, "oof_s": oof_s, "sklearn": sklearn.__version__,
                    "python": platform.python_version(), "machine": "kaggle" if _KAGGLE else "local"},
        "notebook": "case2_random_forest_tu.py",
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
