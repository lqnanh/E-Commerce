
import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import ParameterSampler, PredefinedSplit
from sklearn.pipeline import Pipeline

try:
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows (cp1258) không in được tiếng Việt
except AttributeError:
    pass                                      # stdout của Jupyter/Kaggle đã là UTF-8

# FAST = True: mỗi cây bootstrap 50% dữ liệu (max_samples=0.5), n_estimators chỉ {200, 300}.
# Nhanh hơn ~3 lần. Chỉ bật nếu không đủ thời gian, và phải ghi vào notes (script tự ghi).
FAST = False

IN_NOTEBOOK = "__file__" not in globals()
ON_KAGGLE = Path("/kaggle/input").exists()
if ON_KAGGLE:
    found = sorted(Path("/kaggle/input").rglob("train_folds.csv"))
    assert found, "Không thấy train_folds.csv trong /kaggle/input - đã Add Input dataset chưa?"
    DATA, ROOT = found[0].parent, Path("/kaggle/working")
else:
    ROOT = Path.cwd() if IN_NOTEBOOK else Path(__file__).resolve().parents[1]
    DATA = ROOT / "data"
OUT = ROOT / "results" / "case1" / "random_forest_ngoc"
FIG = ROOT / "shap_figures"
JSON_PATH = ROOT / "results" / "case1" / "random_forest_ngoc.json"
SEED, TARGET = 42, "SCA"
FEATURES = ["add_count", "remove_count", "view_count", "add_to_detail_ratio", "search_count",
            "unique_categories_viewed", "avg_price_bucket", "avg_dwell_time", "session_duration", "hod", "dow"]

PARAM_DIST = {
    "clf__n_estimators": [200, 300, 500],
    "clf__max_depth": [None, 10, 15, 20, 30],
    "clf__min_samples_leaf": [1, 5, 10, 20, 35, 50],
    "clf__min_samples_split": [2, 10, 20],
    "clf__max_features": ["sqrt", 0.3, 0.5, 0.7, 1.0],
    "clf__class_weight": [None, "balanced_subsample"],
}
if FAST:
    PARAM_DIST["clf__n_estimators"] = [200, 300]
    PARAM_DIST["clf__max_samples"] = [0.5]
N_ITER = 30
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)

# PART: cấu hình nào (0..29) chạy trong lần này. None = tất cả. Chia nhiều lần chạy, ví dụ
# PART = range(0, 15) rồi PART = range(15, 30). Cấu hình đã có checkpoint thì tự bỏ qua.
PART = None
# Cấu hình cây rất lớn (min_samples_leaf=1, hoặc max_depth=None với leaf<=5) chạy ít worker hơn để không hết RAM.
# Không ảnh hưởng kết quả, chỉ chậm hơn.
HEAVY_JOBS = 2
CKPT = OUT / "cv_parts"


# RMSE trên XÁC SUẤT P(SCA=1), không trên nhãn 0/1 (CONTRIBUTING mục 5).
def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def load():
    tr = pd.read_csv(DATA / "train_folds.csv")
    te = pd.read_csv(DATA / "test.csv")
    assert len(tr) == 143_771 and len(te) == 70_814, (len(tr), len(te))
    assert sorted(tr["fold"].unique()) == list(range(10))
    assert set(FEATURES) <= set(tr.columns) and set(FEATURES) <= set(te.columns)
    return tr, te


def build():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(random_state=SEED, n_jobs=1))])


def class_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    return {"threshold": float(thr),
            "precision_1": precision_score(y, pred, zero_division=0), "recall_1": recall_score(y, pred),
            "f1_1": f1_score(y, pred),
            "precision_0": precision_score(y, pred, pos_label=0, zero_division=0),
            "recall_0": recall_score(y, pred, pos_label=0), "f1_0": f1_score(y, pred, pos_label=0, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "pr_auc": average_precision_score(y, p), "accuracy": accuracy_score(y, pred)}


def prob_metrics(y, p):
    p = np.clip(p, 0, 1)
    mse = mean_squared_error(y, p)
    return {"rmse": float(np.sqrt(mse)), "mse": float(mse), "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p)), "roc_auc": float(roc_auc_score(y, p))}


def to_py(o):
    if isinstance(o, dict):
        return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    return o.item() if hasattr(o, "item") else o


# ------------------------------------------------------------------ 1. tune
# Tương đương RandomizedSearchCV(n_iter=30, random_state=42, cv=PredefinedSplit, refit theo RMSE):
# cùng 30 cấu hình (ParameterSampler), cùng fold, cùng seed -> cùng kết quả. Khác ở chỗ mỗi cấu hình
# xong là ghi checkpoint (điểm từng fold + xác suất OOF), nên Kaggle ngừng giữa chừng không mất công.
def candidates():
    return list(ParameterSampler(PARAM_DIST, N_ITER, random_state=SEED))


def is_heavy(p):
    leaf = p["clf__min_samples_leaf"]
    return leaf == 1 or (p["clf__max_depth"] is None and leaf <= 5)


def fit_fold(params, X, y, tr_idx, va_idx):
    model = build().set_params(**params)
    t = time.perf_counter()
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    fit_s = time.perf_counter() - t
    return va_idx, model.predict_proba(X.iloc[va_idx])[:, 1], fit_s


def restore_checkpoints():
    """Chép checkpoint của các lần chạy trước (Add Input = output notebook cũ) vào CKPT."""
    CKPT.mkdir(parents=True, exist_ok=True)
    if not ON_KAGGLE:
        return
    for f in Path("/kaggle/input").rglob("cand_*"):
        if f.parent.name == "cv_parts" and not (CKPT / f.name).exists():
            shutil.copy(f, CKPT / f.name)


def save_candidate(i, params, y, fold_out):
    oof = np.full(len(y), np.nan)
    res = {"i": i, "params": params, "rmse": [], "auc": [], "fit_s": []}
    for va, p, fit_s in fold_out:                          # đúng thứ tự fold 0..9
        oof[va] = p
        res["rmse"].append(rmse(y.iloc[va], p))
        res["auc"].append(float(roc_auc_score(y.iloc[va], p)))
        res["fit_s"].append(fit_s)
    np.save(CKPT / f"cand_{i:02d}_oof.npy", oof)
    (CKPT / f"cand_{i:02d}.json").write_text(json.dumps(to_py(res)), encoding="utf-8")   # json = "đã xong"
    return res


def run_batch(ids, cands, X, y, folds, n_jobs, t0):
    """Chạy liên tục mọi (cấu hình, fold) của ids để CPU không ngồi chờ; cấu hình nào đủ 10 fold thì ghi ngay."""
    if not ids:
        return
    tasks = [(i, k) for i in ids for k in range(len(folds))]
    gen = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(fit_fold)(cands[i], X, y, *folds[k]) for i, k in tasks)
    buf = {}
    for (i, k), r in zip(tasks, gen):
        buf.setdefault(i, []).append(r)
        if len(buf[i]) == len(folds):
            res = save_candidate(i, cands[i], y, buf.pop(i))
            print(f"cand {i:02d} xong -> RMSE {np.mean(res['rmse']):.5f}, AUC {np.mean(res['auc']):.4f}, "
                  f"fit TB {np.mean(res['fit_s']) / 60:.1f} phút/fold (đã chạy {(time.perf_counter() - t0) / 60:.0f} phút)"
                  f"  {cands[i]}", flush=True)


def step_tune():
    """Trả về True nếu đã đủ 30 cấu hình và đã refit xong."""
    OUT.mkdir(parents=True, exist_ok=True)
    tr, _ = load()
    X, y = tr[FEATURES], tr[TARGET]
    folds = list(PredefinedSplit(tr["fold"].values).split())
    print(f"Train {X.shape}, SCA=1 = {y.mean():.4f}, NA = {X.isna().sum()[lambda s: s > 0].to_dict()}", flush=True)

    cands = candidates()
    restore_checkpoints()
    todo = [i for i in (PART if PART is not None else range(N_ITER)) if not (CKPT / f"cand_{i:02d}.json").exists()]
    print(f"Đã có checkpoint: {N_ITER - len([i for i in range(N_ITER) if not (CKPT / f'cand_{i:02d}.json').exists()])}"
          f"/{N_ITER}. Lần này chạy: {todo}", flush=True)
    t0 = time.perf_counter()
    run_batch([i for i in todo if not is_heavy(cands[i])], cands, X, y, folds, -1, t0)
    run_batch([i for i in todo if is_heavy(cands[i])], cands, X, y, folds, HEAVY_JOBS, t0)

    missing = [i for i in range(N_ITER) if not (CKPT / f"cand_{i:02d}.json").exists()]
    if missing:
        print(f"Còn thiếu cấu hình {missing}. Chạy tiếp (Add Input output lần này) trước khi refit/test.")
        return False

    res = [json.loads((CKPT / f"cand_{i:02d}.json").read_text(encoding="utf-8")) for i in range(N_ITER)]
    cv_table = pd.DataFrame([{**{k.replace("clf__", ""): str(v) for k, v in r["params"].items()},
                              "mean_rmse": np.mean(r["rmse"]), "std_rmse": np.std(r["rmse"]),
                              "mean_auc": np.mean(r["auc"]), "std_auc": np.std(r["auc"]),
                              "mean_fit_time": np.mean(r["fit_s"]),
                              **{f"split{k}_rmse": v for k, v in enumerate(r["rmse"])},
                              **{f"split{k}_auc": v for k, v in enumerate(r["auc"])}} for r in res])
    cv_table.index.name = "candidate"
    cv_table.to_csv(OUT / "cv_results.csv")
    i = int(np.argmin(cv_table["mean_rmse"].values))       # như refit="rmse" (hoà thì lấy chỉ số nhỏ nhất)
    best = cands[i]
    fold_rmse, fold_auc = np.array(res[i]["rmse"]), np.array(res[i]["auc"])

    # p*: chọn trên xác suất OOF của cấu hình tốt nhất (không bao giờ dùng test)
    oof = np.load(CKPT / f"cand_{i:02d}_oof.npy")
    pd.DataFrame({"fold": tr["fold"], "SCA": y, "p_oof": oof}).to_csv(OUT / "oof_pred.csv", index=False)
    macro = [f1_score(y, oof >= thr, average="macro") for thr in THRESHOLDS]
    p_star = float(THRESHOLDS[int(np.argmax(macro))])

    # refit trên toàn bộ train; n_jobs=-1 chỉ để nhanh, RF có random_state nên kết quả không đổi
    t = time.perf_counter()
    model = build().set_params(**best, clf__n_jobs=-1).fit(X, y)
    refit_s = time.perf_counter() - t
    joblib.dump(model, OUT / "model.joblib", compress=3)

    info = {"best_candidate": i, "best_params": {k.replace("clf__", ""): v for k, v in best.items()},
            "cv_10fold": {"rmse_mean": fold_rmse.mean(), "rmse_std": fold_rmse.std(),
                          "auc_mean": fold_auc.mean(), "auc_std": fold_auc.std()},
            "p_star": p_star, "oof_macro_f1_at_p_star": max(macro),
            "oof_prob_metrics": prob_metrics(y, oof),
            "total_fit_time_s": float(cv_table["mean_fit_time"].sum() * 10), "refit_time_s": refit_s,
            "mean_fit_time_per_fold_s": float(cv_table.loc[i, "mean_fit_time"])}
    (OUT / "tune_info.json").write_text(json.dumps(to_py(info), indent=2), encoding="utf-8")
    print(json.dumps(to_py(info), indent=2), flush=True)
    return True


# ------------------------------------------------------------------ 2. test (một lần)
def step_test(force=False):
    marker = OUT / "TEST_ALREADY_RUN"
    if marker.exists() and not force:
        sys.exit("Tập test đã được dùng. Chỉ chạy lại để tái tạo đúng bảng cũ: thêm --force")
    info = json.loads((OUT / "tune_info.json").read_text(encoding="utf-8"))
    model = joblib.load(OUT / "model.joblib")
    tr, te = load()
    Xte, yte = te[FEATURES], te[TARGET]
    p = model.predict_proba(Xte)[:, 1]
    marker.touch()

    base_p = np.full(len(yte), tr[TARGET].mean())
    metrics_test = prob_metrics(yte, p)
    metrics_class = {"at_0.5": class_metrics(yte, p, 0.5), "at_p_star": class_metrics(yte, p, info["p_star"])}
    baseline = prob_metrics(yte, base_p)
    pd.DataFrame({"SCA": yte, "p_test": p}).to_csv(OUT / "test_pred.csv", index=False)
    pd.DataFrame([{"model": "RF", **metrics_test}, {"model": "Baseline (p = train mean)", **baseline}]) \
        .to_csv(OUT / "test_metrics.csv", index=False)
        
    imp = pd.Series(model.named_steps["clf"].feature_importances_, index=FEATURES, name="impurity_importance")
    imp.sort_values(ascending=False).to_csv(OUT / "feature_importance.csv")

    out = {
        "case": "case1", "model": "random_forest", "author": "ngoc",
        "features": FEATURES, "n_features": len(FEATURES),
        "split": "data/train_folds.csv (143,771) / data/test.csv (70,814), không chia lại, không xoá dòng; "
                 "CV 10-fold theo cột fold (PredefinedSplit); seed 42; test dùng một lần sau tuning",
        "best_params": info["best_params"],
        "metrics_test": metrics_test,
        "metrics_class": metrics_class,
        "cv_10fold": info["cv_10fold"],
        "baseline_test": baseline,
        "notebook": "notebooks/case1_rf_ngoc.py",
        "notes": (f"{'Chạy trên Kaggle. ' if ON_KAGGLE else ''}"
                  f"{'FAST=True: max_samples=0.5, n_estimators {200,300}. ' if FAST else ''}"
                  f"RandomForestClassifier, predict_proba[:,1]. Pipeline: SimpleImputer(median) -> RF. "
                  f"Random search n_iter={N_ITER}: ParameterSampler(random_state=42) + vòng CV 10 fold tự viết "
                  f"(tương đương RandomizedSearchCV, thêm checkpoint từng cấu hình để chạy nhiều phiên Kaggle), "
                  f"chọn theo RMSE trên xác suất trung bình 10 fold, refit trên toàn bộ train. Không gian: "
                  f"{json.dumps({k.replace('clf__', ''): [str(v) for v in vals] for k, vals in PARAM_DIST.items()})}. "
                  f"hod, dow giữ dạng số. p* = {info['p_star']} chọn trên OOF train (tối đa macro-F1). "
                  f"Top importance (impurity): {', '.join(imp.sort_values(ascending=False).index[:5])}. "
                  f"Chi tiết trong results/case1/random_forest_ngoc/."),
    }
    JSON_PATH.write_text(json.dumps(to_py(out), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(to_py(out), indent=2, ensure_ascii=False))


# ------------------------------------------------------------------ 3. SHAP
def step_shap(n=5000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    FIG.mkdir(exist_ok=True)
    model = joblib.load(OUT / "model.joblib")
    tr, _ = load()
    Xs = tr[FEATURES].sample(n, random_state=SEED)
    Xs_imp = pd.DataFrame(model.named_steps["imp"].transform(Xs), columns=FEATURES, index=Xs.index)
    sv = shap.TreeExplainer(model.named_steps["clf"]).shap_values(Xs_imp)
    sv = sv[:, :, 1] if np.ndim(sv) == 3 else sv[1]      # SHAP cho lớp SCA=1 (bỏ giỏ)

    imp = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES, name="mean_abs_shap").sort_values(ascending=False)
    imp.to_csv(FIG / "case1_rf_ngoc_shap_importance.csv")
    for kind, name in [("dot", "summary"), ("bar", "bar")]:
        shap.summary_plot(sv, Xs_imp, plot_type=kind, show=False)
        plt.tight_layout()
        plt.savefig(FIG / f"case1_rf_ngoc_shap_{name}.png", dpi=150)
        plt.close()
    top = imp.index[0]
    shap.dependence_plot(top, sv, Xs_imp, show=False)
    plt.tight_layout()
    plt.savefig(FIG / f"case1_rf_ngoc_shap_dependence_{top}.png", dpi=150)
    plt.close()
    print(imp.round(4).to_string())


if __name__ == "__main__":
    step = "all" if IN_NOTEBOOK else (sys.argv[1] if len(sys.argv) > 1 else "")
    print(f"step={step}  DATA={DATA}  OUT={OUT}  FAST={FAST}", flush=True)
    if step not in ("tune", "test", "shap", "all"):
        sys.exit("Dùng: python notebooks/case1_rf_ngoc.py tune|test|shap|all [--force]")
    tuned = step_tune() if step in ("tune", "all") else True
    if not tuned:
        print("Chưa đủ 30 cấu hình nên dừng ở đây, chưa chạy test và SHAP.")
    else:
        if step in ("test", "all"):
            step_test(force="--force" in sys.argv)
        if step in ("shap", "all"):
            step_shap()
