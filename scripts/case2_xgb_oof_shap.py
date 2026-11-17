"""Tai lap OOF SHAP (log-odds) cho XGBoost Case 2 theo dung code cell 5 + 8 cua
notebooks/case2_xgboost_tran.ipynb, dung best_params trong results/case2/xgboost_tran.json.

Kiem tra: mean|SHAP| phai khop shap_figures/case2_xgb_tran_shap_oof_importance.csv.
Output: shap_figures/case2_xgb_tran_shap_oof.csv.gz, case2_xgb_tran_shap_oof_values_X.csv.gz
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import Pipeline
from xgboost import DMatrix, XGBClassifier

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "shap_figures"
cfg = json.loads((ROOT / "results/case2/xgboost_tran.json").read_text(encoding="utf-8"))
features, bp = cfg["features"], cfg["best_params"]

train = pd.read_csv(ROOT / "data/train_folds.csv")
X, y = train[features], train["SCA"].astype(int)
cv = PredefinedSplit(test_fold=train["fold"].to_numpy())

contrib = np.full((len(X), len(features) + 1), np.nan)
X_imp = np.full((len(X), len(features)), np.nan)
for k, (fit_idx, val_idx) in enumerate(cv.split()):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                random_state=42, n_jobs=1, tree_method="hist", **bp)),
    ])
    pipe.fit(X.iloc[fit_idx], y.iloc[fit_idx])
    xi = pipe.named_steps["imputer"].transform(X.iloc[val_idx])
    m = DMatrix(xi, feature_names=list(features))
    contrib[val_idx] = pipe.named_steps["model"].get_booster().predict(m, pred_contribs=True)
    X_imp[val_idx] = xi
    print(f"fold {k} done", flush=True)

shap_df = pd.DataFrame(contrib[:, :-1], columns=features)
imp = shap_df.abs().mean()
ref = pd.read_csv(FIG / "case2_xgb_tran_shap_oof_importance.csv", index_col=0).iloc[:, 0]
diff = (imp[ref.index] - ref).abs().max()
print(imp.sort_values(ascending=False).round(6).to_string())
print(f"max |diff| vs Tran's importance: {diff:.2e}")
assert diff < 1e-3, "khong khop notebook cua Tran"

shap_df.to_csv(FIG / "case2_xgb_tran_shap_oof.csv.gz", index=False)
pd.DataFrame(X_imp, columns=features).to_csv(FIG / "case2_xgb_tran_shap_oof_values_X.csv.gz", index=False)
print("saved")
