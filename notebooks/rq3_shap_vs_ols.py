"""RQ3 + H1/H2: so sanh giai thich OLS voi SHAP cua cac mo hinh cay.

    python notebooks/rq3_shap_vs_ols.py rf1      # tinh SHAP cho RF Case 1 (Ngoc) tren 5,000 dong train (lau)
    python notebooks/rq3_shap_vs_ols.py analyze  # bang xep hang, bang dau/huong, SHAP theo nhom (nhanh)
    python notebooks/rq3_shap_vs_ols.py plots    # dependence plot cho 11 bien cua RF Case 1

Mau: tr.sample(5000, random_state=42) - CUNG cac dong voi SHAP cua RF Case 1 (Ngoc) va RF Case 2 (Tu).
OLS: SHAP cua mo hinh tuyen tinh = beta_j * (x_j - E[x_j]) (Lundberg & Lee, 2017), E lay tren toan bo train.
     hod (sin, cos) va dow (6 dummy) duoc cong lai thanh 1 gia tri cho moi bien goc truoc khi xep hang.
Huong tac dong cua 1 bien = Spearman(gia tri bien, SHAP cua bien) tren mau: > 0 tang P(SCA=1), < 0 giam.
XGBoost: doc shap_figures/case{1,2}_xgb_*_shap_oof.csv.gz (+ values_X) neu co, lay dung 5,000 dong mau.
Ket qua ghi vao results/rq3/.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
DATA, FIG, OUT = ROOT / "data", ROOT / "shap_figures", ROOT / "results" / "rq3"
SEED, N = 42, 5000
C1 = ["add_count", "remove_count", "view_count", "add_to_detail_ratio", "search_count",
      "unique_categories_viewed", "avg_price_bucket", "avg_dwell_time", "session_duration", "hod", "dow"]
C2 = ["add_count", "remove_count", "view_count", "detail_count", "add_to_detail_ratio", "search_count",
      "unique_categories_viewed", "avg_price_bucket", "max_price_bucket", "unique_products_viewed",
      "avg_dwell_time", "session_duration", "hod", "dow"]
RF1_VALUES = FIG / "case1_rf_ngoc_shap_values.csv"
# Gia thuyet (Muc 2.5): bien -> dau ky vong
HYP = {"H1a": {"view_count": -1, "session_duration": -1},
       "H1b": {"unique_categories_viewed": +1, "unique_products_viewed": +1},
       "H2": {"add_count": +1}}
# Nhom bien tuong quan manh trong Case 2 (Muc 2.3): doc SHAP theo nhom
GROUPS = {"browsing (view_count + detail_count + unique_products_viewed)":
          ["view_count", "detail_count", "unique_products_viewed"],
          "price (avg_price_bucket + max_price_bucket)": ["avg_price_bucket", "max_price_bucket"]}


def sample():
    tr = pd.read_csv(DATA / "train_folds.csv")
    assert len(tr) == 143_771
    return tr, tr.sample(n=N, random_state=SEED)


# ------------------------------------------------------------------ RF Case 1 SHAP
def _chunk(clf, X):
    import shap
    sv = shap.TreeExplainer(clf).shap_values(X, check_additivity=True)
    return sv[1] if isinstance(sv, list) else sv[..., 1]


def step_rf1(n_rows=None):
    import shap
    from joblib import Parallel, delayed
    tr, s = sample()
    if n_rows:
        s = s.iloc[:n_rows]
    model = joblib.load(ROOT / "results" / "case1" / "random_forest_ngoc" / "model.joblib")
    clf = model.named_steps["clf"]
    assert list(clf.classes_) == [0, 1]
    # joblib tao bang sklearn khac -> SimpleImputer.transform loi; dien median thu cong bang statistics_ (tuong duong)
    med = pd.Series(model.named_steps["imp"].statistics_, index=C1)
    Xs = s[C1].fillna(med).astype(float)
    p = clf.predict_proba(Xs.values)[:, 1]
    Xs = Xs.values
    t0 = time.perf_counter()
    chunks = [c for c in np.array_split(np.arange(len(Xs)), 4) if len(c)]
    sv = np.vstack(Parallel(n_jobs=len(chunks))(delayed(_chunk)(clf, Xs[c]) for c in chunks))
    base = float(np.ravel(shap.TreeExplainer(clf).expected_value)[1])
    err = float(np.abs(base + sv.sum(1) - p).max())
    print(f"RF1 SHAP {sv.shape} in {time.perf_counter() - t0:.0f}s | E[p]={base:.6f} | additivity err {err:.1e}")
    assert err < 1e-6
    if n_rows:
        return
    v = pd.DataFrame(sv, columns=[f"shap_{f}" for f in C1], index=s.index)
    v.insert(0, "p", p)
    v.insert(0, "SCA", s.SCA)
    v.insert(0, "fold", s.fold)
    v.rename_axis("row_train_folds").to_csv(RF1_VALUES)
    old = pd.read_csv(FIG / "case1_rf_ngoc_shap_importance.csv", index_col=0).iloc[:, 0]
    new = pd.Series(np.abs(sv).mean(0), index=C1)
    print("max |diff| vs case1_rf_ngoc_shap_importance.csv:", float((old - new[old.index]).abs().max()))


# ------------------------------------------------------------------ OLS "SHAP"
def ols_shap(tr, s):
    coef = pd.read_csv(ROOT / "results" / "case1" / "ols_kiet" / "ols_coefficients.csv", index_col=0)
    b, p = coef["coef"], coef["p_value"]

    def design(df):
        X = df[C1].copy()
        X["avg_price_bucket"] = X["avg_price_bucket"].fillna(tr["avg_price_bucket"].median())
        D = X.drop(columns=["hod", "dow"])
        D["hod_sin"], D["hod_cos"] = np.sin(2 * np.pi * X.hod / 24), np.cos(2 * np.pi * X.hod / 24)
        for k in range(2, 8):
            D[f"dow_{k}"] = (X.dow == k).astype(float)
        return D

    Dtr, Ds = design(tr), design(s)
    cols = [c for c in b.index if c != "const"]
    phi = (Ds[cols] - Dtr[cols].mean()) * b[cols]
    out = pd.DataFrame({f: phi[f] for f in C1 if f in phi})
    out["hod"] = phi[["hod_sin", "hod_cos"]].sum(1)
    out["dow"] = phi[[f"dow_{k}" for k in range(2, 8)]].sum(1)
    # kiem tra: tong phi + E[du doan] = du doan OLS (truoc khi cat ve [0,1])
    pred = b["const"] + (Ds[cols] * b[cols]).sum(1)
    base = b["const"] + (Dtr[cols].mean() * b[cols]).sum()
    assert np.allclose(base + out.sum(1), pred)
    return out[C1], b, p


def load_xgb_oof(prefix, tr, idx):
    """SHAP OOF cua XGBoost (pred_contribs, bo cot bias) + X da dien median; None neu chua co file."""
    sp, xp = FIG / f"{prefix}_shap_oof.csv.gz", FIG / f"{prefix}_shap_oof_values_X.csv.gz"
    if not xp.exists():
        xp = FIG / f"{prefix}_shap_values_X.csv.gz"  # ten file cua Lam
    if not (sp.exists() and xp.exists()):
        return None
    S, X = pd.read_csv(sp), pd.read_csv(xp)
    assert len(S) == len(X) == len(tr) and list(S.columns) == list(X.columns)
    # kiem tra thu tu dong trung train_folds.csv tren cac cot khong thieu
    full = [c for c in X.columns if not tr[c].isna().any()]
    assert np.allclose(X[full].values, tr[full].values, atol=1e-3), f"{prefix}: thu tu dong khac train_folds.csv"
    return S, X


def direction(x, phi):
    r = spearmanr(x, phi).statistic
    return float(r)


def step_analyze():
    OUT.mkdir(parents=True, exist_ok=True)
    tr, s = sample()
    Xs1 = s[C1].fillna(tr[C1].median())
    Xs2 = s[C2].fillna(tr[C2].median())

    ols, beta, pval = ols_shap(tr, s)
    rf1 = pd.read_csv(RF1_VALUES, index_col=0)
    assert (rf1.index.values == s.index.values).all()
    rf1 = rf1[[f"shap_{f}" for f in C1]].set_axis(C1, axis=1)
    rf2 = pd.read_csv(FIG / "case2_rf_tu_shap_values.csv", index_col=0)
    assert (rf2.index.values == s.index.values).all()
    rf2 = rf2[[f"shap_{f}" for f in C2]].set_axis(C2, axis=1)
    xgb1_imp = pd.read_csv(FIG / "case1_xgb_lam_shap_importance.csv", index_col=0).iloc[:, 0]

    # ---- 1. Case 1: xep hang mean|SHAP| (OLS, RF, XGBoost)
    imp1 = pd.DataFrame({"OLS": ols.abs().mean(), "RF": rf1.abs().mean(), "XGBoost": xgb1_imp[C1]})
    rank1 = imp1.rank(ascending=False, method="min").astype(int).add_prefix("rank_")
    t1 = pd.concat([imp1.add_prefix("mean_abs_shap_"), rank1], axis=1).sort_values("rank_OLS")
    t1.to_csv(OUT / "case1_importance_rank.csv")
    rho = {f"{a} vs {b_}": spearmanr(imp1[a], imp1[b_]).statistic
           for a, b_ in [("OLS", "RF"), ("OLS", "XGBoost"), ("RF", "XGBoost")]}
    top5 = {m: list(imp1[m].sort_values(ascending=False).index[:5]) for m in imp1}

    # ---- 2. Huong tac dong (Spearman x vs SHAP); hod/dow khong co thu tu nen bo qua
    num1 = [f for f in C1 if f not in ("hod", "dow")]
    num2 = [f for f in C2 if f not in ("hod", "dow")]
    dir1 = pd.DataFrame({"OLS_beta": beta[num1], "OLS_p": pval[num1],
                         "OLS_dir": [direction(Xs1[f], ols[f]) for f in num1],
                         "RF1_dir": [direction(Xs1[f], rf1[f]) for f in num1]}, index=num1)
    dir2 = pd.Series({f: direction(Xs2[f], rf2[f]) for f in num2}, name="RF2_dir")
    dirs = dir1.join(dir2, how="outer")
    # XGBoost: SHAP OOF (log-odds) tren toan bo train, thu tu dong = train_folds.csv -> lay dung 5,000 dong mau
    xgb_cols = []
    for col, prefix, num in [("XGB1_dir", "case1_xgb_lam", num1), ("XGB2_dir", "case2_xgb_tran", num2)]:
        sv = load_xgb_oof(prefix, tr, s.index)
        if sv is None:
            print(f"!! chua co {prefix}_shap_oof.csv.gz -> bo qua {col}")
            continue
        dirs[col] = pd.Series({f: direction(sv[1].loc[s.index, f], sv[0].loc[s.index, f]) for f in num})
        xgb_cols.append(col)
    dirs = dirs.loc[num2]
    dirs.to_csv(OUT / "direction_by_feature.csv")
    sign_agree = int((np.sign(dirs.OLS_dir) == np.sign(dirs.RF1_dir)).sum())

    # ---- 3. Bang gia thuyet
    labels = [("OLS_dir", "OLS (C1)"), ("RF1_dir", "RF (C1)"), ("RF2_dir", "RF (C2)"),
              ("XGB1_dir", "XGBoost (C1)"), ("XGB2_dir", "XGBoost (C2)")]
    labels = [(c, lab) for c, lab in labels if c in dirs.columns]
    rows = []
    for h, feats in HYP.items():
        for f, exp in feats.items():
            row = {"hypothesis": h, "feature": f, "expected": "+" if exp > 0 else "-"}
            for col, lab in labels:
                v = dirs.loc[f, col] if f in dirs.index else np.nan
                if np.isnan(v):
                    row[lab] = "n/a"
                else:
                    sig = "" if lab != "OLS (C1)" or dirs.loc[f, "OLS_p"] < 0.05 else " (ns)"
                    row[lab] = f"{'+' if v > 0 else '-'} ({v:+.2f}){sig} {'OK' if np.sign(v) == exp else 'X'}"
            rows.append(row)
    hyp = pd.DataFrame(rows)
    hyp.to_csv(OUT / "hypotheses_h1_h2.csv", index=False)

    # ---- 4. Case 2: SHAP theo nhom bien tuong quan (RF Tu)
    grp_rows = []
    for g, fs in GROUPS.items():
        grp_rows.append({"group": g, "mean_abs_group_shap": float(rf2[fs].sum(1).abs().mean()),
                         "sum_of_member_mean_abs": float(rf2[fs].abs().mean().sum()),
                         **{f"mean_abs_{f}": float(rf2[f].abs().mean()) for f in fs}})
    grp = pd.DataFrame(grp_rows)
    rest = [f for f in C2 if not any(f in fs for fs in GROUPS.values())]
    gimp = pd.concat([pd.Series({g: rf2[fs].sum(1).abs().mean() for g, fs in GROUPS.items()}),
                      rf2[rest].abs().mean()]).sort_values(ascending=False)
    grp.to_csv(OUT / "case2_rf_group_shap.csv", index=False)
    gimp.rename("mean_abs_shap").to_csv(OUT / "case2_rf_grouped_importance.csv")

    summary = {"sample": f"train_folds.sample({N}, random_state={SEED})",
               "spearman_case1_importance": rho, "top5_case1": top5,
               "sign_agreement_OLS_vs_RF1": f"{sign_agree}/{len(num1)}",
               "xgboost_direction_from": xgb_cols or "chua co SHAP values -> doc tu beeswarm"}
    for col in xgb_cols:
        summary[f"sign_agreement_OLS_vs_{col[:4]}"] = \
            f"{int((np.sign(dirs.OLS_dir) == np.sign(dirs[col])).loc[num1].sum())}/{len(num1)}"
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("== Case 1 importance (mean|SHAP|; OLS & RF = xac suat, XGBoost = log-odds) ==")
    print(t1.round(4).to_string())
    print("\nSpearman:", {k: round(v, 3) for k, v in rho.items()})
    print("Top 5:", top5)
    print("\n== Huong tac dong: Spearman(x, SHAP) ==")
    print(dirs.round(3).to_string())
    print(f"Dong dau OLS vs RF1: {sign_agree}/{len(num1)}")
    print("\n== H1/H2 ==")
    print(hyp.to_string(index=False))
    print("\n== Case 2 RF: SHAP theo nhom ==")
    print(grp.round(4).to_string(index=False))
    print(gimp.round(4).to_string())


def step_plots():
    """Dependence plot cho moi bien cua RF Case 1 (tu case1_rf_ngoc_shap_values.csv), dat ten giong RF Case 2 (Tu)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    tr, s = sample()
    v = pd.read_csv(RF1_VALUES, index_col=0)
    assert (v.index.values == s.index.values).all()
    sv = v[[f"shap_{f}" for f in C1]].to_numpy()
    med = pd.Series(joblib.load(ROOT / "results" / "case1" / "random_forest_ngoc" / "model.joblib")
                    .named_steps["imp"].statistics_, index=C1)
    Xs = s[C1].fillna(med).astype(float)
    order = np.argsort(-np.abs(sv).mean(0))
    for r, j in enumerate(order, 1):
        f = C1[j]
        shap.dependence_plot(f, sv, Xs, interaction_index="auto", dot_size=8, alpha=0.5, show=False)
        plt.ylabel(f"SHAP của {f} (thang xác suất P(SCA = 1))")
        plt.title(f"RF Case 1 - dependence #{r}: {f} (n = {N:,} dòng train)")
        plt.gcf().savefig(FIG / f"case1_rf_ngoc_shap_dependence_{r:02d}_{f}.png", dpi=200, bbox_inches="tight")
        plt.close("all")
        print(f"{r:02d} {f}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "rf1":
        step_rf1(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif step == "plots":
        step_plots()
    elif step == "analyze":
        step_analyze()
    else:
        sys.exit(__doc__)
