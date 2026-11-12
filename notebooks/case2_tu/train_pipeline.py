"""SCA training pipeline - every team member runs the same steps, in this order.

    python train_pipeline.py features          # 1. feature-set experiment + group ablation (10-fold CV on train only)
    python train_pipeline.py tune [MODEL ...]  # 2. grid search + threshold p* per model (default: all 6 models)
    python train_pipeline.py test [--force]    # 3. pick the final model from CV results, then evaluate ONCE on test
    python train_pipeline.py shap              # 4. TreeSHAP for GBTree (needs: pip install shap matplotlib)

Models: LogReg DT RF GBTree GBReg NB.  All outputs go to results/ next to this file.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", message=".*gblinear.*")  # gblinear is deprecated in xgboost 3.x but still works

DATA = Path(__file__).resolve().parent
OUT = DATA / "results"
MODELS_DIR = OUT / "models"
SEED = 42
CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)  # same folds for every model and experiment

# feature groups of section 3.3.1 - order = order of the incremental experiment
GROUPS = {
    "Cart":     ["add_count", "remove_count", "unique_products_in_cart"],
    "Browsing": ["view_count", "detail_count", "add_to_detail_ratio"],
    "Search":   ["search_count", "unique_categories_viewed"],
    "Price":    ["avg_price_bucket", "max_price_bucket"],
    "Session":  ["nevents", "unique_products_viewed", "avg_dwell_time", "session_duration"],
    "Temporal": ["hod", "dow"],
}
CATEGORICAL = ["hod", "dow"]
FEATURE_MODELS = ["GBTree", "LogReg"]  # fast models used for the feature experiment
SELECTION_MODEL = "GBTree"             # the feature set is chosen from this model's results

PARAM_GRIDS = {
    "LogReg": {"clf__C": [0.01, 0.1, 1, 10]},
    "DT":     {"clf__max_depth": [3, 5, 8, 12], "clf__min_samples_leaf": [1, 20, 100]},
    "RF":     {"clf__max_features": ["sqrt", 0.5, 0.9], "clf__min_samples_leaf": [1, 5, 35]},
    "GBTree": {"clf__n_estimators": [150, 300], "clf__max_depth": [3, 5, 7], "clf__learning_rate": [0.05, 0.1, 0.3]},
    "GBReg":  {"clf__reg_lambda": [0, 0.1, 1], "clf__reg_alpha": [0, 0.1, 1]},
    "NB":     {"clf__var_smoothing": [1e-9, 1e-7, 1e-5]},
}
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)


def load(split):
    df = pd.read_csv(DATA / f"{split}.csv")
    return df.drop(columns="SCA"), df.SCA


def build(name, cols):
    """Preprocessing + classifier in one Pipeline, so every CV fold refits the preprocessing on its own train part."""
    cat = [c for c in cols if c in CATEGORICAL]
    num = [c for c in cols if c not in CATEGORICAL]
    impute = SimpleImputer(strategy="median")
    linear = make_column_transformer((make_pipeline(impute, StandardScaler()), num),
                                     (OneHotEncoder(handle_unknown="ignore"), cat))
    if name == "LogReg":
        prep, clf = linear, LogisticRegression(max_iter=1000)
    elif name == "DT":
        prep, clf = impute, DecisionTreeClassifier(random_state=SEED)
    elif name == "RF":
        prep, clf = impute, RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=SEED)
    elif name == "GBTree":  # xgboost handles NaN itself
        prep, clf = "passthrough", XGBClassifier(n_estimators=150, max_depth=3, colsample_bytree=0.8,
                                                 n_jobs=1, random_state=SEED)
    elif name == "GBReg":   # boosting with LINEAR base learners (Rausch et al.'s best model), not trees
        prep, clf = linear, XGBClassifier(booster="gblinear", learning_rate=0.3, n_estimators=150,
                                          n_jobs=1, random_state=SEED)
    elif name == "NB":
        prep = make_column_transformer((make_pipeline(impute, FunctionTransformer(np.log1p)), num),
                                       remainder="passthrough")
        clf = GaussianNB()
    else:
        raise ValueError(f"unknown model {name}, choose from {list(PARAM_GRIDS)}")
    return Pipeline([("prep", prep), ("clf", clf)])


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": threshold, "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "accuracy": accuracy_score(y, pred), "precision_1": precision_score(y, pred, zero_division=0),
            "recall_1": recall_score(y, pred), "f1_1": f1_score(y, pred),
            "f1_0": f1_score(y, pred, pos_label=0, zero_division=0), "macro_f1": f1_score(y, pred, average="macro"),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def save_json(obj, path):
    path.write_text(json.dumps(obj, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)))


# ---------------------------------------------------------------- step 1
def step_features():
    X, y = load("train")
    sets, cols = {}, []
    for i, (group, feats) in enumerate(GROUPS.items(), 1):          # incremental: S1 -> S6 (S6 = all 16)
        cols = cols + feats
        sets[f"S{i} +{group}"] = cols
    for group, feats in GROUPS.items():                             # ablation: all 16 minus one group
        sets[f"All -{group}"] = [c for c in X.columns if c not in feats]
    c1 = pd.read_csv(OUT / "case1_feature_filter_pearson.csv")      # Case 1: multicollinearity filter
    sets["Case1 filter"] = c1.loc[c1.step == "final", "reason"].iloc[0].split(",")
    c2 = pd.read_csv(OUT / "case2_feature_filter.csv")              # Case 2: drop X with corr(X, SCA) ~ 0
    sets["Case2 filter"] = [c for c in X.columns if c in set(c2.loc[~c2["drop"], "var"])]

    rows = []
    for set_name, cols in sets.items():
        for m in FEATURE_MODELS:
            t = time.perf_counter()
            auc = cross_val_score(build(m, cols), X[cols], y, cv=CV, scoring="roc_auc", n_jobs=-1)
            rows.append({"set": set_name, "model": m, "n_features": len(cols), "auc_mean": auc.mean(),
                         "auc_std": auc.std(), "time_s": time.perf_counter() - t, "features": ",".join(cols)})
            print(f"{set_name:14s} {m:7s} {len(cols):2d} vars  AUC {auc.mean():.4f} +/- {auc.std():.4f}")

    res = pd.DataFrame(rows)
    full = res[res["set"].str.startswith("S6")].set_index("model").auc_mean
    res["delta_vs_all16"] = res.auc_mean - res.model.map(full)       # ablation: how much AUC is lost without a group
    res.to_csv(OUT / "feature_experiment.csv", index=False)

    # one-standard-error rule: smallest set whose AUC is within 1 std of the best set
    sel = res[res.model == SELECTION_MODEL]
    best = sel.loc[sel.auc_mean.idxmax()]
    ok = sel[sel.auc_mean >= best.auc_mean - best.auc_std].sort_values(["n_features", "auc_mean"],
                                                                       ascending=[True, False])
    chosen = ok.iloc[0]
    save_json({"set": chosen["set"], "features": chosen["features"].split(","), "auc_mean": chosen["auc_mean"],
               "best_set": best["set"], "best_auc_mean": best["auc_mean"], "best_auc_std": best["auc_std"],
               "rule": f"smallest set with {SELECTION_MODEL} CV AUC >= best - 1 std"},
              OUT / "selected_features.json")
    print(f"\nselected: {chosen['set']} ({chosen['n_features']} vars, AUC {chosen['auc_mean']:.4f}); "
          f"best was {best['set']} (AUC {best['auc_mean']:.4f} +/- {best['auc_std']:.4f})")
    print(f"saved {OUT / 'feature_experiment.csv'} and {OUT / 'selected_features.json'}")


# ---------------------------------------------------------------- step 2
def step_tune(models):
    sel_path = OUT / "selected_features.json"
    if not sel_path.exists():
        sys.exit("run 'python train_pipeline.py features' first (or copy results/selected_features.json from the team)")
    feats = json.loads(sel_path.read_text())["features"]
    X, y = load("train")
    X = X[feats]

    for m in models:
        t = time.perf_counter()
        gs = GridSearchCV(build(m, feats), PARAM_GRIDS[m], cv=CV, scoring="roc_auc", n_jobs=-1)
        gs.fit(X, y)  # refits the best configuration on the whole train set
        tuning_s = time.perf_counter() - t
        i = gs.best_index_

        # threshold p*: out-of-fold probabilities of the best configuration (never the test set)
        oof = cross_val_predict(gs.best_estimator_, X, y, cv=CV, method="predict_proba", n_jobs=-1)[:, 1]
        macro = [f1_score(y, oof >= thr, average="macro") for thr in THRESHOLDS]
        p_star = float(THRESHOLDS[int(np.argmax(macro))])

        info = {"model": m, "features": feats, "best_params": gs.best_params_,
                "cv_auc_mean": gs.best_score_, "cv_auc_std": gs.cv_results_["std_test_score"][i],
                "p_star": p_star, "oof_macro_f1_at_p_star": max(macro),
                "oof_macro_f1_at_0.5": f1_score(y, oof >= 0.5, average="macro"),
                "fit_time_per_fold_s": gs.cv_results_["mean_fit_time"][i],
                "n_configs": len(gs.cv_results_["params"]), "tuning_time_s": tuning_s}
        joblib.dump(gs.best_estimator_, MODELS_DIR / f"{m}.joblib")
        save_json(info, MODELS_DIR / f"{m}.json")
        print(f"{m:7s} CV AUC {gs.best_score_:.4f} +/- {info['cv_auc_std']:.4f}  p*={p_star:.2f}  "
              f"tuning {tuning_s:.0f}s  best {gs.best_params_}")


# ---------------------------------------------------------------- step 3
def step_test(force):
    infos = [json.loads(p.read_text()) for p in sorted(MODELS_DIR.glob("*.json"))]
    if not infos:
        sys.exit("no tuned models in results/models - run the 'tune' step first")
    cv = pd.DataFrame(infos).set_index("model")
    missing = set(PARAM_GRIDS) - set(cv.index)
    if missing and not force:  # the test set may be used only once, so wait until all 6 models are tuned
        sys.exit(f"models not tuned yet: {sorted(missing)}. Tune them first (or add --force to test without them)")

    # 1) choose the final model from CV results ONLY, before the test set is touched:
    #    models within 1 std of the best CV AUC are "equally good" -> take the fastest one (Rausch et al. trade-off)
    best = cv.cv_auc_mean.idxmax()
    cands = cv[cv.cv_auc_mean >= cv.loc[best, "cv_auc_mean"] - cv.loc[best, "cv_auc_std"]]
    final = cands.fit_time_per_fold_s.idxmin()
    save_json({"final_model": final, "best_cv_auc_model": best, "candidates_within_1_std": list(cands.index),
               "rule": "among models with CV AUC >= best - 1 std, the one with the lowest fit time"},
              OUT / "final_model.json")
    print(f"final model (chosen on CV only): {final}   [best CV AUC: {best}; candidates: {list(cands.index)}]")

    # 2) evaluate every model ONCE on the test set (for the comparison table) - never tune after this
    marker = OUT / "TEST_ALREADY_RUN"
    if marker.exists() and not force:
        sys.exit("the test set was already used. Re-running only to reproduce the same table? add --force")
    X_test, y_test = load("test")
    rows = [{"model": "Baseline (all = 1)", "setting": "-", **metrics(y_test, np.ones(len(y_test)), 0.5)}]
    for m, info in cv.iterrows():
        model = joblib.load(MODELS_DIR / f"{m}.joblib")
        p = model.predict_proba(X_test[info.features])[:, 1]
        for setting, thr in [("A (0.5)", 0.5), ("B (p*)", info.p_star)]:
            rows.append({"model": m, "setting": setting, **metrics(y_test, p, thr),
                         "cv_auc_mean": info.cv_auc_mean, "fit_time_per_fold_s": info.fit_time_per_fold_s,
                         "final": m == final})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "test_results.csv", index=False)
    marker.touch()
    cols = ["model", "setting", "threshold", "roc_auc", "pr_auc", "accuracy", "precision_1", "recall_1",
            "f1_1", "f1_0", "macro_f1"]
    print(res[cols].round(4).to_string(index=False))
    print(f"saved {OUT / 'test_results.csv'}")


# ---------------------------------------------------------------- step 4
def step_shap():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    info = json.loads((MODELS_DIR / "GBTree.json").read_text())
    model = joblib.load(MODELS_DIR / "GBTree.joblib")
    X, _ = load("train")  # explain on a train sample: interpretation needs no test data
    Xs = X[info["features"]].sample(10000, random_state=SEED)
    sv = shap.TreeExplainer(model.named_steps["clf"]).shap_values(Xs)  # > 0 pushes towards SCA = 1 (abandon)

    pd.Series(np.abs(sv).mean(axis=0), index=Xs.columns, name="mean_abs_shap") \
        .sort_values(ascending=False).to_csv(OUT / "shap_importance.csv")
    for kind, fname in [("dot", "shap_summary.png"), ("bar", "shap_importance.png")]:
        shap.summary_plot(sv, Xs, plot_type=kind, show=False)
        plt.tight_layout()
        plt.savefig(OUT / fname, dpi=150)
        plt.close()
    print(f"saved shap_importance.csv, shap_summary.png, shap_importance.png in {OUT}")


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    step, args = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if step == "features":
        step_features()
    elif step == "tune":
        unknown = set(args) - set(PARAM_GRIDS)
        if unknown:
            sys.exit(f"unknown model(s) {sorted(unknown)}; choose from {list(PARAM_GRIDS)}")
        step_tune(args or list(PARAM_GRIDS))
    elif step == "test":
        step_test(force="--force" in args)
    elif step == "shap":
        step_shap()
    else:
        sys.exit(__doc__)
