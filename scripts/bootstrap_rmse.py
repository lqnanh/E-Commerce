"""Paired bootstrap on test RMSE (report Section 4.6).

1. Re-fit the three models whose test predictions were not saved, using the exact pipelines and
   best_params already recorded in their JSON (OLS Case 1, XGBoost Case 1, RF Case 2), and check
   that the re-fitted test RMSE/AUC match the JSON. RF Case 1 and XGBoost Case 2 predictions are
   read from the files their authors saved.
2. Paired bootstrap: B = 2000 resamples of the 70,814 test sessions (seed 42), same rows for all
   models. Difference = RMSE(model A) - RMSE(model B); negative = A has lower error.
   95% and Bonferroni 98.75% (4 comparisons) percentile intervals.

Run from repo root:  python scripts/bootstrap_rmse.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "bootstrap"
SEED, B, N_COMP = 42, 2000, 4
TOL = 1e-6

CASE1 = ["add_count", "remove_count", "view_count", "add_to_detail_ratio", "search_count",
         "unique_categories_viewed", "avg_price_bucket", "avg_dwell_time", "session_duration", "hod", "dow"]
CASE2 = ["add_count", "remove_count", "view_count", "detail_count", "add_to_detail_ratio", "search_count",
         "unique_categories_viewed", "avg_price_bucket", "max_price_bucket", "unique_products_viewed",
         "avg_dwell_time", "session_duration", "hod", "dow"]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def json_metrics(rel):
    return json.loads((ROOT / "results" / rel).read_text(encoding="utf-8"))["metrics_test"]


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def check(name, y, p, ref):
    r, a = rmse(y, p), float(roc_auc_score(y, p))
    ok = abs(r - ref["rmse"]) < TOL and abs(a - ref["roc_auc"]) < TOL
    print(f"{name:10s} RMSE {r:.6f} (json {ref['rmse']:.6f})  AUC {a:.6f} (json {ref['roc_auc']:.6f})  "
          f"{'OK' if ok else 'MISMATCH'}", flush=True)
    return ok


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "data" / "train_folds.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    y, yt = train["SCA"].astype(int), test["SCA"].astype(int).to_numpy()
    preds, ok = {}, {}

    # --- OLS Case 1 (Kiet): same pipeline as scripts/case1_ols_kiet.py, no tuning
    ols = load_module(ROOT / "scripts" / "case1_ols_kiet.py", "ols_kiet")
    m = ols.make_pipeline().fit(train[CASE1], y)
    preds["OLS_c1"] = ols.predict_proba1(m, test[CASE1])
    ok["OLS_c1"] = check("OLS_c1", yt, preds["OLS_c1"], json_metrics("case1/ols_kiet.json"))

    # --- XGBoost Case 1 (Lam): same pipeline as notebooks/case1_xgb_lam.py + best_params
    xgbm = load_module(ROOT / "notebooks" / "case1_xgb_lam.py", "xgb_lam")
    bp = json.loads((ROOT / "results/case1/xgboost_lam.json").read_text(encoding="utf-8"))["best_params"]
    m = xgbm.make_pipeline({f"model__{k}": v for k, v in bp.items()}).fit(train[CASE1], y)
    preds["XGB_c1"] = np.clip(m.predict_proba(test[CASE1])[:, 1], 0, 1)
    ok["XGB_c1"] = check("XGB_c1", yt, preds["XGB_c1"], json_metrics("case1/xgboost_lam.json"))

    # --- RF Case 2 (Tu): same pipeline as notebooks/case2_random_forest_tu.py + best_params
    bp = json.loads((ROOT / "results/case2/random_forest_tu.json").read_text(encoding="utf-8"))["best_params"]
    m = Pipeline([("imp", SimpleImputer(strategy="median")),
                  ("clf", RandomForestClassifier(n_jobs=-1, **bp))]).fit(train[CASE2], y)
    preds["RF_c2"] = m.predict_proba(test[CASE2])[:, 1]
    ok["RF_c2"] = check("RF_c2", yt, preds["RF_c2"], json_metrics("case2/random_forest_tu.json"))

    # --- saved predictions: RF Case 1 (Ngoc), XGBoost Case 2 (Tran)
    rf1 = pd.read_csv(ROOT / "results/case1/random_forest_ngoc/test_pred.csv")
    xg2 = pd.read_csv(ROOT / "results/case2/xgboost_tran_test_pred.csv").sort_values("row_id")
    assert (rf1["SCA"].to_numpy() == yt).all() and (xg2["SCA"].to_numpy() == yt).all(), "row order differs"
    preds["RF_c1"], preds["XGB_c2"] = rf1["p_test"].to_numpy(), xg2["p_test"].to_numpy()
    ok["RF_c1"] = check("RF_c1", yt, preds["RF_c1"], json_metrics("case1/random_forest_ngoc.json"))
    ok["XGB_c2"] = check("XGB_c2", yt, preds["XGB_c2"], json_metrics("case2/xgboost_tran.json"))

    pd.DataFrame({"SCA": yt, **{k: preds[k] for k in ("OLS_c1", "XGB_c1", "RF_c2")}}).to_csv(
        OUT / "test_pred_refit.csv", index_label="row_id")

    # --- paired bootstrap on RMSE
    comps = [("RF_c1", "OLS_c1"), ("XGB_c1", "OLS_c1"), ("RF_c2", "RF_c1"), ("XGB_c2", "XGB_c1")]
    sq = {k: (yt - np.asarray(p, dtype=float)) ** 2 for k, p in preds.items()}
    rng = np.random.default_rng(SEED)
    n = len(yt)
    diffs = {c: np.empty(B) for c in comps}
    for b in range(B):
        idx = rng.integers(0, n, n)
        r = {k: np.sqrt(v[idx].mean()) for k, v in sq.items()}
        for a, c in comps:
            diffs[(a, c)][b] = r[a] - r[c]

    alpha_b = 0.05 / N_COMP
    rows = []
    for a, c in comps:
        d = diffs[(a, c)]
        point = np.sqrt(sq[a].mean()) - np.sqrt(sq[c].mean())
        rows.append({"comparison": f"{a} - {c}", "rmse_a": np.sqrt(sq[a].mean()), "rmse_b": np.sqrt(sq[c].mean()),
                     "diff": point, "se_boot": d.std(ddof=1),
                     "ci95_low": np.percentile(d, 2.5), "ci95_high": np.percentile(d, 97.5),
                     "ci9875_low": np.percentile(d, 100 * alpha_b / 2),
                     "ci9875_high": np.percentile(d, 100 * (1 - alpha_b / 2)),
                     "excludes_zero_9875": bool(np.percentile(d, 100 * alpha_b / 2) > 0
                                                or np.percentile(d, 100 * (1 - alpha_b / 2)) < 0)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "rmse_bootstrap.csv", index=False)
    (OUT / "rmse_bootstrap.json").write_text(json.dumps(
        {"B": B, "seed": SEED, "n_test": int(n), "bonferroni_level": 1 - alpha_b,
         "diff_definition": "RMSE(a) - RMSE(b); negative means a has lower test RMSE",
         "refit_matches_json": ok, "results": res.to_dict(orient="records")}, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.float_format", "{:.5f}".format):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
