"""Case 2 - Random Forest (author: tu): giai thich mo hinh cuoi bang SHAP (TreeExplainer).

    python case2_random_forest_shap_tu.py            # 5,000 dong train -> results/contrib/random_forest/shap/
    python case2_random_forest_shap_tu.py --quick    # chay thu 200 dong -> results/contrib_quick/random_forest/shap/
    python case2_random_forest_shap_tu.py --refit    # bo qua joblib, fit lai RF voi BEST tren toan bo train
    Trong repo nhom (notebooks/): doc data/train_folds.csv, ghi shap_figures/case2_rf_tu_shap_*.

Mo hinh: random_forest_tu.joblib = Pipeline(SimpleImputer(median) + RF) da refit tren toan bo 143,771 dong train
boi case2_random_forest_tu.py. Neu khong tim thay joblib, hoac joblib tao bang phien ban sklearn khac (ban goc: 1.6.1),
script fit lai RF voi dung BEST + random_state=42 tren toan bo train va doi chieu feature_importances_ voi JSON.

Mau: 5,000 dong train_folds.csv, random_state=42. KHONG dung test.csv.
SHAP lay cho lop SCA = 1 tren thang xac suat: expected_value + tong SHAP cua 1 dong = P(SCA=1) cua RF.
SHAP > 0: bien day du doan ve phia bo gio (SCA = 1); SHAP < 0: day ve phia mua.
Can: pip install shap matplotlib
"""
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

try:
    import shap
except ImportError:
    raise SystemExit("Thieu thu vien shap: pip install shap matplotlib")

_INPUT = Path("/kaggle/input")
_KAGGLE = sorted(_INPUT.glob("**/train_folds.csv")) if _INPUT.exists() else []
HERE = Path(__file__).resolve().parent
# Repo nhom (E-Commerce): script o notebooks/, du lieu o data/, hinh o shap_figures/
REPO = HERE.parent if not _KAGGLE and (HERE.parent / "shap_figures").is_dir() else None
if _KAGGLE:
    DATA = _KAGGLE[0].parent  # Kaggle: dataset ncamtu06/sca-folds
elif REPO is not None:
    DATA = REPO / "data"
else:
    DATA = HERE
RESULTS = Path("/kaggle/working") if _KAGGLE else DATA / "results"
SEED = 42
QUICK = "--quick" in sys.argv
REFIT = "--refit" in sys.argv
N_SAMPLE = 200 if QUICK else 5_000
N_JOBS = max(1, min(4, os.cpu_count() or 1))  # moi tien trinh giu 1 ban sao RF trong RAM
CASE2 = ["add_count", "remove_count", "view_count", "detail_count", "add_to_detail_ratio", "search_count",
         "unique_categories_viewed", "avg_price_bucket", "max_price_bucket", "unique_products_viewed",
         "avg_dwell_time", "session_duration", "hod", "dow"]
NAME = "random_forest_tu"
# best_params cua lan chay Kaggle 2026-11-14 (results/contrib/random_forest/random_forest_tu.json)
BEST = {"n_estimators": 200, "max_features": 0.3, "min_samples_leaf": 20, "class_weight": None, "random_state": SEED}
if REPO is not None and not QUICK:
    # Quy uoc repo nhom: shap_figures/case2_rf_tu_shap_*.png, dependence de phang cung thu muc
    OUT, PREFIX, DEP_DIR = REPO / "shap_figures", "case2_rf_tu", None
else:
    OUT, PREFIX, DEP_DIR = RESULTS / ("contrib_quick" if QUICK else "contrib") / "random_forest" / "shap", NAME, "dependence"


def find(name):
    """Tim file trong Kaggle input (output cua notebook RF duoc them lam input), repo nhom, roi thu muc results local."""
    cands = sorted(_INPUT.glob(f"**/{name}")) if _INPUT.exists() else []
    if REPO is not None:
        cands.append(REPO / "results" / "case2" / name)  # JSON nop nhom (joblib khong commit -> se refit)
    cands.append(DATA / "results" / "contrib" / "random_forest" / name)
    return next((p for p in cands if p.exists()), None)


def load_model(X, y):
    path = None if REFIT else find(f"{NAME}.joblib")
    if path is not None:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = joblib.load(path)
            mismatch = [w for w in caught if type(w.message).__name__ == "InconsistentVersionWarning"]
            if not mismatch:
                return model, f"joblib: {path.name} (sklearn {sklearn.__version__}, khong train lai)"
            print(f"!! joblib tao bang sklearn khac (dang dung {sklearn.__version__}): {mismatch[0].message}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"!! khong load duoc {path}: {e!r}", flush=True)
        print("-> fit lai RF voi BEST tren toan bo train", flush=True)
    elif not REFIT:
        print(f"!! khong tim thay {NAME}.joblib -> fit lai RF voi BEST tren toan bo train", flush=True)
    params = {**BEST, "n_estimators": 20} if QUICK else BEST  # --quick: 20 cay cho nhanh (chi de chay thu)
    model = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("clf", RandomForestClassifier(**params, n_jobs=-1))])
    model.fit(X, y)
    return model, f"refit tren toan bo train voi {params}"


def shap_chunk(clf, X):
    # Khong truyen background -> feature_perturbation="tree_path_dependent"; model_output="raw" cua RF sklearn
    # chinh la xac suat (trung binh xac suat cua cac cay), nen SHAP nam tren thang xac suat.
    ex = shap.TreeExplainer(clf)
    sv = ex.shap_values(X, check_additivity=True)
    sv = sv[1] if isinstance(sv, list) else sv[..., 1]  # shap cu: list theo lop; shap moi: (n, f, lop)
    return sv, float(np.ravel(ex.expected_value)[1])


def save(fig_path):
    plt.gcf().savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close("all")


def main():
    t0 = time.perf_counter()
    tr = pd.read_csv(DATA / "train_folds.csv")
    assert len(tr) == 143_771 and sorted(tr.fold.unique()) == list(range(10))
    X, y = tr[CASE2], tr.SCA
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"RF case2 SHAP: train {X.shape}, mau {N_SAMPLE} | sklearn {sklearn.__version__} shap {shap.__version__}",
          flush=True)

    model, source = load_model(X, y)
    clf = model.named_steps["clf"]
    assert list(model.feature_names_in_) == CASE2, "thu tu cot khong khop luc train"
    assert list(clf.classes_) == [0, 1]
    if not QUICK:
        got = {k: clf.get_params()[k] for k in BEST}
        assert got == BEST, f"tham so mo hinh {got} khac BEST {BEST}"
    print(f"mo hinh: {source}", flush=True)

    imp_ref, imp_diff = None, None  # doi chieu voi feature_importance trong JSON cua lan chay chinh
    js = find(f"{NAME}.json")
    if js is not None:
        imp_ref = json.loads(js.read_text(encoding="utf-8"))["feature_importance"]
        imp_diff = float(max(abs(clf.feature_importances_[i] - imp_ref[f]) for i, f in enumerate(CASE2)))
        print(f"feature_importances_ lech toi da so voi JSON: {imp_diff:.2e}", flush=True)
        if imp_diff > 1e-9 and not QUICK:
            print("!! mo hinh KHAC mo hinh trong JSON (sklearn khac?) - ghi ro trong bao cao", flush=True)

    sample = tr.sample(n=N_SAMPLE, random_state=SEED)
    # SHAP giai thich RF tren dau vao DA dien median (median cua toan bo train, tu imputer cua Pipeline)
    Xs = pd.DataFrame(model.named_steps["imp"].transform(sample[CASE2]), columns=CASE2, index=sample.index)
    p = model.predict_proba(sample[CASE2])[:, 1]

    t1 = time.perf_counter()
    chunks = [c for c in np.array_split(np.arange(len(Xs)), N_JOBS) if len(c)]
    parts = Parallel(n_jobs=len(chunks))(delayed(shap_chunk)(clf, Xs.iloc[c]) for c in chunks)
    sv = np.vstack([s for s, _ in parts])
    base = parts[0][1]
    shap_s = time.perf_counter() - t1
    add_err = float(np.max(np.abs(base + sv.sum(axis=1) - p)))
    assert add_err < 1e-6, f"SHAP khong cong lai bang P(SCA=1): lech {add_err}"
    print(f"SHAP {sv.shape} xong sau {shap_s / 60:.1f} phut | E[P(SCA=1)] = {base:.6f} | lech cong {add_err:.1e}",
          flush=True)

    imp = pd.DataFrame({"feature": CASE2,
                        "mean_abs_shap": np.abs(sv).mean(axis=0),
                        "mean_shap": sv.mean(axis=0),
                        "impurity_importance": clf.feature_importances_})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp.insert(0, "rank", np.arange(1, len(imp) + 1))
    imp["rank_impurity"] = imp.impurity_importance.rank(ascending=False, method="min").astype(int)
    imp.to_csv(OUT / f"{PREFIX}_shap_importance.csv", index=False)

    vals = pd.DataFrame(sv, columns=[f"shap_{f}" for f in CASE2], index=sample.index)
    vals.insert(0, "p", p)
    vals.insert(0, "SCA", sample.SCA)
    vals.insert(0, "fold", sample.fold)
    vals.rename_axis("row_train_folds").to_csv(OUT / f"{PREFIX}_shap_values.csv")

    xlabel = "Giá trị SHAP (tác động lên P(SCA = 1))"
    shap.summary_plot(sv, Xs, max_display=len(CASE2), show=False)
    plt.xlabel(xlabel)
    plt.title(f"SHAP summary - Random Forest Case 2 (n = {N_SAMPLE:,} dòng train)")
    save(OUT / f"{PREFIX}_shap_summary.png")

    shap.summary_plot(sv, Xs, plot_type="bar", max_display=len(CASE2), show=False)
    plt.xlabel("mean(|SHAP|) (thang xác suất P(SCA = 1))")
    plt.title("Mức độ quan trọng của biến theo SHAP - Random Forest Case 2")
    save(OUT / f"{PREFIX}_shap_bar.png")

    dep = OUT / DEP_DIR if DEP_DIR else OUT
    dep.mkdir(exist_ok=True)
    dep_name = f"{PREFIX}_dependence" if DEP_DIR else f"{PREFIX}_shap_dependence"
    for r, f in zip(imp["rank"], imp.feature):
        shap.dependence_plot(f, sv, Xs, interaction_index="auto", dot_size=8, alpha=0.5, show=False)
        plt.ylabel(f"SHAP của {f}")
        plt.title(f"Dependence #{r}: {f}")
        save(dep / f"{dep_name}_{r:02d}_{f}.png")

    summary = {
        "case": "case2", "model": "random_forest", "author": "tu",
        "explainer": "shap.TreeExplainer(feature_perturbation='tree_path_dependent', model_output='raw' = xac suat)",
        "target_class": "SCA = 1 (bo gio)", "scale": "xac suat P(SCA=1)",
        "model_source": source, "best_params": BEST,
        "sample": {"from": "train_folds.csv", "n": N_SAMPLE, "random_state": SEED,
                   "sca_rate": float(sample.SCA.mean()), "input": "da dien median (imputer cua Pipeline)"},
        "expected_value": base, "mean_p_sample": float(p.mean()), "additivity_max_abs_err": add_err,
        "impurity_vs_json_max_abs_diff": imp_diff,
        "mean_abs_shap": dict(zip(imp.feature, imp.mean_abs_shap.astype(float))),
        "mean_shap": dict(zip(imp.feature, imp.mean_shap.astype(float))),
        "runtime": {"shap_s": shap_s, "total_s": time.perf_counter() - t0, "n_jobs": len(chunks),
                    "sklearn": sklearn.__version__, "shap": shap.__version__,
                    "python": platform.python_version(), "machine": "kaggle" if _KAGGLE else "local"},
        "notebook": "case2_random_forest_shap_tu.py",
    }
    (OUT / f"{PREFIX}_shap.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(imp.to_string(index=False))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
