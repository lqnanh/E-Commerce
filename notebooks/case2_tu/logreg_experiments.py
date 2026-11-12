"""Logistic Regression tuning: ways to let a linear model capture the non-linear patterns RF finds (train only, never test).

    python logreg_experiments.py      # round 1: 5 preprocessings x 2 feature sets -> results/logreg_tuning/
    python logreg_experiments.py 2    # round 2: best preprocessings, wider C grid, spline + interactions together
    python logreg_experiments.py 3    # round 3: + the 4 cart-churn features of RF Lan 4 (27 vars, same set as RF)

Every variant = preprocessing of the numeric columns + one-hot hod/dow + LogisticRegression, grid over C,
same 10 folds as train_pipeline / train_folds.csv. Variant "base" reproduces the Kaggle run sca-logreg-case2.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.preprocessing import (FunctionTransformer, KBinsDiscretizer, OneHotEncoder, PolynomialFeatures,
                                   SplineTransformer, StandardScaler)

from feature_engineering import CASE2, DERIVED, RAWF
from train_pipeline import CV, OUT, THRESHOLDS, load, metrics, save_json

CAT = ["hod", "dow"]
NEW = [c for c in DERIVED + RAWF if c != "net_cart"]  # same 23-feature set as RF Lan 3 (net_cart = add - remove)
slog = FunctionTransformer(lambda x: np.sign(x) * np.log1p(np.abs(x)), feature_names_out="one-to-one")  # signed log


def numeric(kind):
    if kind == "spline+inter":  # both at once: per-variable curves + pairwise interactions
        return make_union(numeric("spline"), numeric("log+inter"))
    imp = SimpleImputer(strategy="median")
    return {"base":   make_pipeline(imp, StandardScaler()),
            "log":    make_pipeline(imp, slog, StandardScaler()),
            "bins":   make_pipeline(imp, KBinsDiscretizer(n_bins=10, strategy="quantile", encode="onehot",
                                                          quantile_method="averaged_inverted_cdf")),
            "spline": make_pipeline(imp, slog, SplineTransformer(n_knots=6, degree=3)),
            "log+inter": make_pipeline(imp, slog, StandardScaler(),
                                       PolynomialFeatures(2, interaction_only=True, include_bias=False),
                                       StandardScaler())}[kind]


def model(kind, cols):
    prep = make_column_transformer((numeric(kind), [c for c in cols if c not in CAT]),
                                   (OneHotEncoder(handle_unknown="ignore"), CAT))
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=3000))])


ROUND = sys.argv[1] if len(sys.argv) > 1 else "1"
if ROUND == "3":
    from rf_lan4 import CHURN, load_with_churn
    X, y, *_ = load_with_churn()
else:
    X, y = load("train_fe")
FEATURE_SETS = {"Case2": CASE2, "Case2+new": CASE2 + NEW}
if ROUND == "3":
    FEATURE_SETS["Case2+new+churn"] = CASE2 + NEW + CHURN
folds = [va for _, va in CV.split(X, y)]
if ROUND == "1":
    runs, C = [(k, fs) for fs in ["Case2", "Case2+new"] for k in ["base", "log", "bins", "spline", "log+inter"]],               [0.001, 0.01, 0.1, 1]
elif ROUND == "2":  # C = 1 was the grid edge for spline and log+inter in round 1
    runs, C = [(k, fs) for fs in ["Case2", "Case2+new"] for k in ["spline", "log+inter", "spline+inter"]], [0.1, 1, 10, 100]
else:  # round 2: AUC flat for C >= 1
    runs, C = [("spline+inter", "Case2+new+churn")], [10, 100]
out = OUT / "logreg_tuning"
out.mkdir(exist_ok=True)
sfx = "" if ROUND == "1" else f"_r{ROUND}"
rows, fold_auc = [], {}
if ROUND != "1":  # compare against round 1 "base | Case2" (= Kaggle run) on the same folds
    fold_auc["base | Case2"] = pd.read_csv(out / "logreg_fold_auc.csv")["base | Case2"].to_numpy()
for kind, fs in runs:
    cols = FEATURE_SETS[fs]
    gs = GridSearchCV(model(kind, cols), {"clf__C": C}, cv=CV, scoring="roc_auc", n_jobs=4)
    gs.fit(X[cols], y)
    oof = cross_val_predict(gs.best_estimator_, X[cols], y, cv=CV, method="predict_proba", n_jobs=4)[:, 1]
    auc = np.array([roc_auc_score(y.iloc[va], oof[va]) for va in folds])
    macro = [f1_score(y, oof >= t, average="macro") for t in THRESHOLDS]
    p_star = float(THRESHOLDS[int(np.argmax(macro))])
    m = metrics(y, oof, p_star)
    name = f"{kind} | {fs}"
    fold_auc[name] = auc
    rows.append({"variant": name, "n_features": len(cols), "best_C": gs.best_params_["clf__C"],
                 "auc_mean": auc.mean(), "auc_std": auc.std(), "p_star": p_star, "macro_f1": m["macro_f1"],
                 "f1_0": m["f1_0"], "recall_0": float(m["tn"] / (m["tn"] + m["fp"])),
                 "fit_time_per_fold_s": gs.cv_results_["mean_fit_time"][gs.best_index_],
                 "grid_auc": json.dumps({str(c): round(s, 4) for c, s in zip(gs.cv_results_["param_clf__C"],
                                                                             gs.cv_results_["mean_test_score"])})})
    base = fold_auc["base | Case2"]
    rows[-1].update(delta_vs_base=auc.mean() - base.mean(), folds_better=int((auc > base).sum()))
    print(f"{name:22s} C={rows[-1]['best_C']:<6} AUC {auc.mean():.4f} +/- {auc.std():.4f}  "
          f"d={rows[-1]['delta_vs_base']:+.4f} ({rows[-1]['folds_better']}/10)  p*={p_star:.2f}  "
          f"macroF1 {m['macro_f1']:.4f}  {rows[-1]['grid_auc']}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(out / f"logreg_variants{sfx}.csv", index=False)
pd.DataFrame(fold_auc).to_csv(out / f"logreg_fold_auc{sfx}.csv", index_label="fold")
print(res.drop(columns="grid_auc").round(4).to_string(index=False))
