"""New session features + CV check of whether they help. Shared files (train.csv, test.csv, preprocess.py) are untouched.

    python feature_engineering.py build   # raw Coveo CSVs -> train_fe.csv / test_fe.csv (same rows, same order, + new columns)
    python feature_engineering.py cv      # 10-fold CV on train_fe.csv: Case 2 set vs Case 2 + new features (never test)
    python feature_engineering.py tune    # Case 2 filter on the new features, then RF grid search -> results/rf_fe/

New feature groups (all computed on events BEFORE the first purchase, same ex-ante cut as preprocess.py):
  Derived  ratios of existing columns
  Raw      start/cart-anchored: time and browsing before the first add, prices of carted vs viewed products
  End      anchored at the end of the session (activity after the last add). Kept separate because for buyers the
           session is cut just before the purchase - if this group alone gives a big jump, suspect leakage.
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split

DATA = Path(__file__).resolve().parent
RAW = DATA.parent / "train"

CASE2 = ["add_count", "remove_count", "view_count", "detail_count", "add_to_detail_ratio", "search_count",
         "unique_categories_viewed", "avg_price_bucket", "max_price_bucket", "unique_products_viewed",
         "avg_dwell_time", "session_duration", "hod", "dow"]
DERIVED = ["remove_to_add", "net_cart", "views_per_min", "detail_revisit"]
RAWF = ["time_to_first_add", "details_before_first_add", "min_cart_price", "cart_price_range",
        "avg_viewed_price", "cart_vs_view_price"]
END = ["time_after_last_add", "events_after_last_add"]


def step_build():
    con = duckdb.connect()
    con.execute("SET memory_limit='9GB'")
    browsing = (f"read_csv('{(RAW / 'browsing_train.csv').as_posix()}', header=true, columns={{'session_id_hash': 'VARCHAR', "
                "'event_type': 'VARCHAR', 'product_action': 'VARCHAR', 'product_sku_hash': 'VARCHAR', "
                "'server_timestamp_epoch_ms': 'BIGINT', 'hashed_url': 'VARCHAR'}, max_line_size=20000000)")
    sku = (f"read_csv('{(RAW / 'sku_to_content.csv').as_posix()}', header=true, columns={{'product_sku_hash': 'VARCHAR', "
           "'description_vector': 'VARCHAR', 'category_hash': 'VARCHAR', 'image_vector': 'VARCHAR', "
           "'price_bucket': 'DOUBLE'}, max_line_size=20000000)")
    # same stage 1/2 as preprocess.py: cart sessions, distinct events, cut at the first purchase
    con.execute(f"""
    CREATE TEMP TABLE ev AS
    SELECT e.*, k.price_bucket
    FROM (SELECT DISTINCT session_id_hash, event_type, product_action, product_sku_hash, server_timestamp_epoch_ms AS ts
          FROM {browsing}
          WHERE session_id_hash IN (SELECT session_id_hash FROM {browsing} WHERE product_action = 'add')) e
    LEFT JOIN {sku} k USING (product_sku_hash)""")
    con.execute("""
    CREATE TEMP TABLE ev_cut AS
    SELECT ev.* FROM ev JOIN (SELECT session_id_hash, min(ts) FILTER (WHERE product_action = 'purchase') AS fp
                              FROM ev GROUP BY 1) l USING (session_id_hash)
    WHERE l.fp IS NULL OR ev.ts < l.fp""")
    new = con.sql("""
    WITH s AS (
        SELECT session_id_hash, min(ts) AS t0, max(ts) AS t1,
               min(ts) FILTER (WHERE product_action = 'add') AS first_add,
               max(ts) FILTER (WHERE product_action = 'add') AS last_add
        FROM ev_cut GROUP BY 1)
    SELECT session_id_hash,
        (any_value(first_add) - any_value(t0)) / 1000.0                                  AS time_to_first_add,
        count(*) FILTER (WHERE product_action = 'detail' AND ts < first_add)            AS details_before_first_add,
        min(price_bucket) FILTER (WHERE product_action = 'add')                          AS min_cart_price,
        max(price_bucket) FILTER (WHERE product_action = 'add')
          - min(price_bucket) FILTER (WHERE product_action = 'add')                      AS cart_price_range,
        avg(price_bucket) FILTER (WHERE product_action = 'detail')                       AS avg_viewed_price,
        (any_value(t1) - any_value(last_add)) / 1000.0                                   AS time_after_last_add,
        count(*) FILTER (WHERE ts > last_add)                                            AS events_after_last_add
    FROM ev_cut JOIN s USING (session_id_hash)
    GROUP BY session_id_hash""").df()

    df = pd.read_csv(DATA / "sca_dataset.csv")
    df = df.merge(new, on="session_id_hash", how="left", validate="one_to_one")  # left merge keeps row order
    assert df.time_to_first_add.notna().all() and (df.time_to_first_add >= 0).all()
    df["cart_vs_view_price"] = df.avg_price_bucket - df.avg_viewed_price
    df["remove_to_add"] = df.remove_count / df.add_count            # add_count > 0 in every row
    df["net_cart"] = df.add_count - df.remove_count
    df["views_per_min"] = df.view_count / (df.session_duration / 60 + 1)
    df["detail_revisit"] = df.detail_count / df.unique_products_viewed.clip(lower=1)

    # identical call to split.py -> identical rows in identical order; checked against the shared train.csv
    train, test = train_test_split(df, test_size=0.33, stratify=df.SCA, random_state=42)
    shared = pd.read_csv(DATA / "train.csv")
    pd.testing.assert_frame_equal(train[shared.columns].reset_index(drop=True), shared, check_dtype=False)
    for name, d in [("train_fe", train), ("test_fe", test)]:
        d.drop(columns="session_id_hash").to_csv(DATA / f"{name}.csv", index=False)
        print(f"saved {name}.csv {d.shape[0]:,} rows, +{len(DERIVED + RAWF + END)} new columns")
    print(train[DERIVED + RAWF + END + ["SCA"]].describe().T.round(3).to_string())


def step_cv():
    from train_pipeline import CV, OUT, THRESHOLDS, build, load
    X, y = load("train_fe")
    sets = {"Case2 (14)": CASE2, "Case2 + Derived + Raw": CASE2 + DERIVED + RAWF,
            "Case2 + Derived + Raw + End": CASE2 + DERIVED + RAWF + END}
    models = {"RF": {"clf__max_features": 0.5, "clf__min_samples_leaf": 35},  # best params of the Kaggle grid
              "GBTree": {}}                                                   # pipeline default, not tuned
    folds = [va for _, va in CV.split(X, y)]
    rows = []
    for m, params in models.items():
        for name, cols in sets.items():
            # one CV pass: out-of-fold probabilities give both the per-fold AUC and the threshold p*
            oof = cross_val_predict(build(m, cols).set_params(**params), X[cols], y, cv=CV,
                                    method="predict_proba", n_jobs=-1)[:, 1]
            auc = np.array([roc_auc_score(y.iloc[va], oof[va]) for va in folds])
            macro = [f1_score(y, oof >= t, average="macro") for t in THRESHOLDS]
            rows.append({"model": m, "set": name, "n_features": len(cols), "auc_mean": auc.mean(),
                         "auc_std": auc.std(), "p_star": THRESHOLDS[int(np.argmax(macro))],
                         "macro_f1_at_p_star": max(macro), **{f"fold{k}": a for k, a in enumerate(auc)}})
            print(f"{m:6s} {name:28s} {len(cols):2d} vars  AUC {auc.mean():.4f} +/- {auc.std():.4f}  "
                  f"p*={rows[-1]['p_star']:.2f}  macro-F1 {max(macro):.4f}", flush=True)
    res = pd.DataFrame(rows)
    fold_cols = [f"fold{k}" for k in range(len(folds))]
    for m in models:  # paired comparison: same folds, so compare fold by fold
        r = res[res.model == m].set_index("set")
        for name in list(sets)[1:]:
            d = r.loc[name, fold_cols].to_numpy(float) - r.loc["Case2 (14)", fold_cols].to_numpy(float)
            res.loc[(res.model == m) & (res.set == name), ["delta_auc", "folds_better"]] = d.mean(), (d > 0).sum()
    res.to_csv(OUT / "fe_experiment.csv", index=False)

    cols = CASE2 + DERIVED + RAWF + END  # importances of the full RF, to spot a feature that dominates (leak sign)
    rf = build("RF", cols).set_params(**models["RF"], clf__n_jobs=-1).fit(X[cols], y)
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, cols, name="importance").sort_values(ascending=False)
    imp.to_csv(OUT / "fe_importance.csv")
    print(res[["model", "set", "auc_mean", "delta_auc", "folds_better", "macro_f1_at_p_star"]].round(4).to_string())
    print(imp.round(4).to_string())


def step_tune():
    """Lan 3: Case 2 filter on the new features, then grid search RF on the cleaned set (train only)."""
    import json
    import time

    import joblib
    from scipy.stats import spearmanr
    from sklearn.model_selection import GridSearchCV
    from train_pipeline import CV, OUT, THRESHOLDS, build, load, metrics, save_json
    X, y = load("train_fe")
    out = OUT / "rf_fe"
    out.mkdir(exist_ok=True)

    # same rule as case2_feature_filter.py: drop if Spearman p > 0.05; drop exact linear combinations (like nevents)
    rows = []
    for c in DERIVED + RAWF:
        m = X[c].notna()
        r, p = spearmanr(X[c][m], y[m])
        reason = "add_count - remove_count: exact linear combination" if c == "net_cart" else \
                 (f"Spearman p > 0.05" if p > 0.05 else "")
        rows.append({"var": c, "spearman": r, "p_value": p, "drop": bool(reason), "reason": reason})
    filt = pd.DataFrame(rows)
    filt.to_csv(out / "new_feature_filter.csv", index=False)
    feats = CASE2 + filt.loc[~filt["drop"], "var"].tolist()
    print(filt.round(4).to_string(index=False), f"\n-> {len(feats)} features", flush=True)

    grid = {"clf__n_estimators": [500], "clf__min_samples_leaf": [35, 100, 300], "clf__max_features": ["sqrt", 0.3, 0.5]}
    t = time.perf_counter()
    gs = GridSearchCV(build("RF", feats), grid, cv=CV, scoring="roc_auc", n_jobs=12, verbose=1).fit(X[feats], y)
    tuning_s = time.perf_counter() - t
    i = gs.best_index_
    oof = cross_val_predict(gs.best_estimator_, X[feats], y, cv=CV, method="predict_proba", n_jobs=10)[:, 1]
    macro = [f1_score(y, oof >= t, average="macro") for t in THRESHOLDS]
    p_star = float(THRESHOLDS[int(np.argmax(macro))])

    fold_auc = np.array([gs.cv_results_[f"split{k}_test_score"][i] for k in range(10)])
    prev = pd.read_csv(OUT / "fe_experiment.csv").query("model == 'RF'").set_index("set")
    fold_cols = [f"fold{k}" for k in range(10)]
    vs = {name: {"delta_auc": float((fold_auc - prev.loc[name, fold_cols].to_numpy(float)).mean()),
                 "folds_better": int((fold_auc > prev.loc[name, fold_cols].to_numpy(float)).sum())}
          for name in ["Case2 (14)", "Case2 + Derived + Raw"]}
    info = {"model": "RF", "features": feats, "best_params": gs.best_params_, "cv_auc_mean": gs.best_score_,
            "cv_auc_std": gs.cv_results_["std_test_score"][i], "fold_auc": fold_auc.tolist(), "p_star": p_star,
            "oof_at_p_star": metrics(y, oof, p_star), "oof_at_0.5": metrics(y, oof, 0.5), "vs": vs,
            "fit_time_per_fold_s": gs.cv_results_["mean_fit_time"][i], "tuning_time_s": tuning_s}
    save_json(info, out / "RF_fe.json")
    pd.DataFrame(gs.cv_results_).drop(columns="params").to_csv(out / "RF_fe_grid.csv", index=False)
    pd.Series(gs.best_estimator_.named_steps["clf"].feature_importances_, feats, name="importance") \
        .sort_values(ascending=False).to_csv(out / "RF_fe_importance.csv")
    joblib.dump(gs.best_estimator_, out / "RF_fe.joblib", compress=3)
    print(pd.DataFrame(gs.cv_results_)[["param_clf__max_features", "param_clf__min_samples_leaf", "mean_test_score",
                                        "std_test_score", "mean_fit_time"]].round(4).to_string(index=False))
    print(json.dumps({k: v for k, v in info.items() if k != "features"}, indent=2, default=str))


if __name__ == "__main__":
    {"build": step_build, "cv": step_cv, "tune": step_tune}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
