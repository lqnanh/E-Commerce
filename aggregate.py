"""Gộp mọi file JSON trong results/ thành bảng so sánh."""
import json
from pathlib import Path

import pandas as pd

rows = []
for p in sorted(Path("results").rglob("*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(d, dict) or "metrics_test" not in d:
        continue  # bỏ qua file phụ (tune_info.json, checkpoint CV...)
    m = d["metrics_test"]
    rows.append({
        "case": d.get("case"),
        "model": d.get("model"),
        "author": d.get("author"),
        "n_features": d.get("n_features"),
        "RMSE": m.get("rmse"),
        "MSE": m.get("mse"),
        "MAE": m.get("mae"),
        "R2": m.get("r2"),
        "file": str(p),
    })

if not rows:
    raise SystemExit("Chưa có file kết quả nào trong results/")

df = pd.DataFrame(rows)

# Mức cải thiện RMSE so với baseline OLS (Case 1)
ols = df[(df.case == "case1") & (df.model.str.lower() == "ols")]
if not ols.empty:
    base = ols["RMSE"].iloc[0]
    df["RMSE_improve_vs_OLS_%"] = ((base - df["RMSE"]) / base * 100).round(2)

df = df.sort_values(["case", "RMSE"])
df.to_csv("summary.csv", index=False)
Path("summary.md").write_text(df.drop(columns=["file"]).to_markdown(index=False), encoding="utf-8")
print(df.drop(columns=["file"]).to_string(index=False))
