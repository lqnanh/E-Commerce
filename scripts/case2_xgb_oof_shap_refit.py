"""Re-compute out-of-fold SHAP values for XGBoost Case 2 (Tran) on the local machine.

Tran's notebook (notebooks/case2_xgboost_tran.ipynb, cell 5 + 8) wrote the per-session OOF SHAP files only to
/kaggle/working and they were not committed. This script repeats the same procedure locally:
  - same pipeline: SimpleImputer(median) -> XGBClassifier(binary:logistic, logloss, hist, n_jobs=1, seed 42)
    with Tran's best_params from results/case2/xgboost_tran.json;
  - for each of the 10 shared folds: fit on the other 9 folds, explain the validation fold with
    booster.predict(pred_contribs=True) (log-odds, bias column dropped), check additivity.
The re-fit is close to but not bit-identical with Tran's Kaggle run, so the files are saved under a
different prefix (case2_xgb_refit_*), not under Tran's names.

Run from repo root:  python scripts/case2_xgb_oof_shap_refit.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import DMatrix, XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "shap_figures"
PREFIX = "case2_xgb_refit_shap_oof"
FEATURES = ["add_count", "remove_count", "view_count", "detail_count", "add_to_detail_ratio", "search_count",
            "unique_categories_viewed", "avg_price_bucket", "max_price_bucket", "unique_products_viewed",
            "avg_dwell_time", "session_duration", "hod", "dow"]


def make_model(params):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42,
                                n_jobs=1, tree_method="hist", **params)),
    ])


def run_fold(model, X, y, fit_idx, val_idx):
    m = clone(model).fit(X.iloc[fit_idx], y.iloc[fit_idx])
    Xv = m.named_steps["imputer"].transform(X.iloc[val_idx])
    booster = m.named_steps["model"].get_booster()
    dm = DMatrix(Xv, feature_names=FEATURES)
    contrib = booster.predict(dm, pred_contribs=True)
    margin = booster.predict(dm, output_margin=True)
    np.testing.assert_allclose(contrib.sum(1), margin, rtol=1e-5, atol=1e-4)
    proba = m.predict_proba(X.iloc[val_idx])[:, 1]
    rmse = float(np.sqrt(mean_squared_error(y.iloc[val_idx], proba)))
    auc = float(roc_auc_score(y.iloc[val_idx], proba))
    return val_idx, Xv, contrib, proba, rmse, auc


def main():
    ref = json.loads((ROOT / "results/case2/xgboost_tran.json").read_text(encoding="utf-8"))
    train = pd.read_csv(ROOT / "data" / "train_folds.csv")
    X, y, folds = train[FEATURES], train["SCA"].astype(int), train["fold"].to_numpy()
    model = make_model(ref["best_params"])

    splits = [(np.where(folds != k)[0], np.where(folds == k)[0]) for k in range(10)]
    out = Parallel(n_jobs=10)(delayed(run_fold)(model, X, y, f, v) for f, v in splits)

    n = len(train)
    shap_oof = np.full((n, len(FEATURES)), np.nan)
    x_oof = np.full((n, len(FEATURES)), np.nan)
    p_oof = np.full(n, np.nan)
    for val_idx, Xv, contrib, proba, _, _ in out:
        shap_oof[val_idx], x_oof[val_idx], p_oof[val_idx] = contrib[:, :-1], Xv, proba
    assert np.isfinite(shap_oof).all() and np.isfinite(p_oof).all()

    rmse_f = np.array([o[4] for o in out])
    auc_f = np.array([o[5] for o in out])
    thr = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y, (p_oof >= t).astype(int), average="macro", zero_division=0) for t in thr]
    check = {
        "cv_rmse_mean_refit": float(rmse_f.mean()), "cv_rmse_mean_tran": ref["cv_10fold"]["rmse_mean"],
        "cv_auc_mean_refit": float(auc_f.mean()), "cv_auc_mean_tran": ref["cv_10fold"]["auc_mean"],
        "p_star_refit": float(thr[int(np.argmax(f1s))]), "p_star_tran": ref["metrics_class"]["at_p_star"]["threshold"],
    }

    pd.DataFrame(shap_oof, columns=FEATURES).to_csv(FIG / f"{PREFIX}.csv.gz", index=False)
    pd.DataFrame(x_oof, columns=FEATURES).to_csv(FIG / f"{PREFIX}_values_X.csv.gz", index=False)
    imp = pd.Series(np.abs(shap_oof).mean(0), index=FEATURES).sort_values(ascending=False)
    imp.rename_axis("feature").rename("mean_abs_shap").to_csv(FIG / f"{PREFIX}_importance.csv")
    tran_imp = pd.read_csv(FIG / "case2_xgb_tran_shap_oof_importance.csv", index_col=0).iloc[:, 0]
    check["spearman_importance_vs_tran_oof"] = float(imp.rank().corr(tran_imp[imp.index].rank()))
    check["max_abs_importance_diff_vs_tran_oof"] = float((imp - tran_imp[imp.index]).abs().max())
    (FIG / f"{PREFIX}_check.json").write_text(json.dumps(
        {"source": "re-fit of Tran's XGBoost Case 2 (best_params from results/case2/xgboost_tran.json)",
         "note": "close to but not bit-identical with Tran's Kaggle run", **check}, indent=2), encoding="utf-8")
    print(json.dumps(check, indent=2))
    print(pd.DataFrame({"refit": imp, "tran_oof": tran_imp[imp.index]}).round(4).to_string())


if __name__ == "__main__":
    main()
