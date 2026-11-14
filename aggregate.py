"""Gộp mọi file JSON trong results/ thành bảng so sánh (theo quy ước trong CONTRIBUTING.md)."""
import json
from pathlib import Path

import pandas as pd

rows = []
for p in sorted(Path("results").rglob("*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    m = d.get("metrics_test", {})
    cv = d.get("cv_10fold", {})          # file chưa theo quy ước (vd. cv_5fold) sẽ để trống
    rows.append({
        "case": d.get("case"),
        "model": d.get("model"),
        "author": d.get("author"),
        "n_features": d.get("n_features"),
        "RMSE": m.get("rmse"),
        "ROC_AUC": m.get("roc_auc"),
        "MSE": m.get("mse"),
        "MAE": m.get("mae"),
        "R2": m.get("r2"),
        "CV_RMSE_mean": cv.get("rmse_mean"),
        "CV_RMSE_std": cv.get("rmse_std"),
        "CV_AUC_mean": cv.get("auc_mean"),
        "CV_AUC_std": cv.get("auc_std"),
        "file": str(p),
    })

if not rows:
    raise SystemExit("Chưa có file kết quả nào trong results/")

df = pd.DataFrame(rows)

# Mức cải thiện RMSE so với OLS (Case 1)
ols = df[(df.case == "case1") & (df.model.str.lower() == "ols")]
if not ols.empty:
    base = ols["RMSE"].iloc[0]
    df["RMSE_improve_vs_OLS_%"] = ((base - df["RMSE"]) / base * 100).round(2)

# Xếp hạng: RMSE là tiêu chí chính, ROC-AUC là tiêu chí phụ (CONTRIBUTING mục 6.1)
df = df.sort_values(["case", "RMSE", "ROC_AUC"], ascending=[True, True, False])
df.to_csv("summary.csv", index=False)
Path("summary.md").write_text(df.drop(columns=["file"]).to_markdown(index=False), encoding="utf-8")
print(df.drop(columns=["file"]).to_string(index=False))
