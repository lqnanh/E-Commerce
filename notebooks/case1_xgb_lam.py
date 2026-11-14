from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train_folds.csv"
TEST = ROOT / "data" / "test.csv"
OUT = ROOT / "results" / "case1"
SHAP_DIR = ROOT / "shap_figures"
TARGET = "SCA"
FOLD = "fold"
SEED = 42
FEATURES = [
    "add_count", "remove_count", "view_count", "add_to_detail_ratio",
    "search_count", "unique_categories_viewed", "avg_price_bucket",
    "avg_dwell_time", "session_duration", "hod", "dow",
]


def neg_probability_rmse(estimator, X, y):
    # XGBClassifier.predict returns hard labels; the project metric is on P(SCA=1).
    p = estimator.predict_proba(X)[:, 1]
    return -float(np.sqrt(mean_squared_error(y, p)))


def probability_metrics(y, p):
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    y = np.asarray(y, dtype=int)
    mse = mean_squared_error(y, p)
    return {
        "rmse": float(np.sqrt(mse)),
        "mse": float(mse),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
    }


def class_metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    labels = (np.asarray(p) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision_1": float(precision_score(y, labels, pos_label=1, zero_division=0)),
        "recall_1": float(recall_score(y, labels, pos_label=1, zero_division=0)),
        "f1_1": float(f1_score(y, labels, pos_label=1, zero_division=0)),
        "precision_0": float(precision_score(y, labels, pos_label=0, zero_division=0)),
        "recall_0": float(recall_score(y, labels, pos_label=0, zero_division=0)),
        "f1_0": float(f1_score(y, labels, pos_label=0, zero_division=0)),
        "balanced_accuracy": float((recall_score(y, labels, pos_label=1, zero_division=0)
                                     + recall_score(y, labels, pos_label=0, zero_division=0)) / 2),
        "macro_f1": float(f1_score(y, labels, average="macro", zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, labels)),
    }


def select_macro_f1_threshold(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)[::-1]
    sorted_p, sorted_y = p[order], y[order]
    ends = np.r_[np.flatnonzero(sorted_p[:-1] != sorted_p[1:]), len(p) - 1]
    tp = np.cumsum(sorted_y)[ends]
    predicted_positive = ends + 1
    fp = predicted_positive - tp
    fn = y.sum() - tp
    tn = len(y) - y.sum() - fp
    f1_1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) != 0)
    f1_0 = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn, dtype=float), where=(2 * tn + fp + fn) != 0)
    macro_f1 = (f1_1 + f1_0) / 2
    return float(sorted_p[ends[int(np.argmax(macro_f1))]])


def make_pipeline(params=None):
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=2,
        verbosity=0,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ]).set_params(**(params or {}))


def main():
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    assert len(train) == 143_771 and len(test) == 70_814, (
        f"Unexpected data sizes: train={len(train):,}, test={len(test):,}"
    )
    assert TARGET in train and FOLD in train and TARGET in test
    assert set(train[FOLD].astype(int).unique()) == set(range(10)), "Expected shared fold IDs 0-9."
    assert set(FEATURES).issubset(train.columns) and set(FEATURES).issubset(test.columns)
    assert list(train[FEATURES].columns) == list(test[FEATURES].columns)
    assert not train[TARGET].isna().any() and not test[TARGET].isna().any()

    X, y = train[FEATURES], train[TARGET].astype(int)
    X_test, y_test = test[FEATURES], test[TARGET].astype(int)
    fold_ids = train[FOLD].astype(int).to_numpy()
    cv = PredefinedSplit(fold_ids)
    assert cv.get_n_splits() == 10

    search_space = {
        "model__n_estimators": [100, 200, 300, 400, 500],
        "model__max_depth": [2, 3, 4, 5, 6],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "model__reg_lambda": [0.5, 1, 2, 5],
        "model__scale_pos_weight": [0.25, 0.5, 1.0],
    }
    search = RandomizedSearchCV(
        make_pipeline(),
        param_distributions=search_space,
        n_iter=30,
        random_state=SEED,
        cv=cv,
        scoring={"rmse": neg_probability_rmse, "auc": "roc_auc"},
        n_jobs=1,
        refit="rmse",
        return_train_score=False,
        verbose=1,
    )
    search.fit(X, y)
    best_params = {key.removeprefix("model__"): value
                   for key, value in search.best_params_.items()}
    best_pipe_params = {f"model__{key}": value for key, value in best_params.items()}

    oof_probability = np.full(len(train), np.nan, dtype=float)
    oof_shap = np.full((len(train), len(FEATURES)), np.nan, dtype=np.float32)
    oof_features = np.full((len(train), len(FEATURES)), np.nan, dtype=np.float32)
    fold_rmse, fold_auc = [], []
    for fold in range(10):
        valid = np.flatnonzero(fold_ids == fold)
        fit = np.flatnonzero(fold_ids != fold)
        pipe = make_pipeline(best_pipe_params)
        pipe.fit(X.iloc[fit], y.iloc[fit])
        p = pipe.predict_proba(X.iloc[valid])[:, 1]
        imputed_valid = pipe.named_steps["imputer"].transform(X.iloc[valid])
        booster = pipe.named_steps["model"]
        contrib = booster.get_booster().predict(
            __import__("xgboost").DMatrix(imputed_valid, feature_names=FEATURES),
            pred_contribs=True,
        )
        oof_probability[valid] = p
        oof_features[valid] = imputed_valid
        oof_shap[valid] = contrib[:, :-1]
        fold_rmse.append(float(np.sqrt(mean_squared_error(y.iloc[valid], p))))
        fold_auc.append(float(roc_auc_score(y.iloc[valid], p)))
        print(f"OOF fold {fold + 1}/10 complete", flush=True)

    assert np.isfinite(oof_probability).all() and np.isfinite(oof_shap).all()
    search_rmse = np.array([
        -search.cv_results_[f"split{i}_test_rmse"][search.best_index_]
        for i in range(10)
    ])
    search_auc = np.array([
        search.cv_results_[f"split{i}_test_auc"][search.best_index_]
        for i in range(10)
    ])
    assert np.allclose(search_rmse, fold_rmse, atol=1e-6), "OOF RMSE differs from tuned CV scores."
    assert np.allclose(search_auc, fold_auc, atol=1e-6), "OOF AUC differs from tuned CV scores."

    p_star = select_macro_f1_threshold(y.to_numpy(), oof_probability)
    cv_result = {
        "rmse_mean": float(np.mean(search_rmse)),
        "rmse_std": float(np.std(search_rmse, ddof=0)),
        "auc_mean": float(np.mean(search_auc)),
        "auc_std": float(np.std(search_auc, ddof=0)),
    }

    final_model = make_pipeline(best_pipe_params)
    final_model.fit(X, y)
    p_test = np.clip(final_model.predict_proba(X_test)[:, 1], 0, 1)
    metrics_test = probability_metrics(y_test, p_test)
    metrics_class = {
        "at_0.5": class_metrics(y_test, p_test, 0.5),
        "at_p_star": class_metrics(y_test, p_test, p_star),
    }

    base_probability = float(y.mean())
    baseline = probability_metrics(y_test, np.full(len(y_test), base_probability))
    overlap_columns = [column for column in train.columns if column in test.columns]
    overlap_rows = test[overlap_columns].merge(
        train[overlap_columns].drop_duplicates(), on=overlap_columns, how="inner"
    ).shape[0]

    OUT.mkdir(parents=True, exist_ok=True)
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(search.cv_results_).to_csv(OUT / "xgboost_lam_cv_10fold.csv", index=False)
    pd.DataFrame([metrics_test]).to_csv(OUT / "xgboost_lam_test_metrics.csv", index=False)
    pd.DataFrame({"p_oof": oof_probability}).to_csv(
        OUT / "xgboost_lam_oof_predictions.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(oof_shap, columns=FEATURES).to_csv(
        SHAP_DIR / "case1_xgb_lam_shap_oof.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(oof_features, columns=FEATURES).to_csv(
        SHAP_DIR / "case1_xgb_lam_shap_values_X.csv.gz", index=False, compression="gzip"
    )

    import runpy
    runpy.run_path(str(ROOT / "notebooks" / "plot_case1_xgb_lam_shap.py"), run_name="__main__")

    result = {
        "case": "case1",
        "model": "xgboost",
        "author": "Lam",
        "features": FEATURES,
        "n_features": len(FEATURES),
        "split": (
            "Provided fixed split: data/train_folds.csv (143771 rows; shared folds 0-9) "
            "and data/test.csv (70814 rows); no rows removed or split regenerated. "
            "RandomizedSearchCV seed=42."
        ),
        "best_params": best_params,
        "metrics_test": metrics_test,
        "metrics_class": metrics_class,
        "cv_10fold": cv_result,
        "baseline_test": baseline,
        "notebook": "notebooks/case1_xgb_lam.ipynb",
        "notes": (
            "Tuning: RandomizedSearchCV, n_iter=30, random_state=42, "
            "PredefinedSplit(train['fold']), scoring={'rmse': probability RMSE, 'auc': roc_auc}, refit='rmse'. "
            "Parameter distributions: n_estimators=[100,200,300,400,500], "
            "max_depth=[2,3,4,5,6], learning_rate=[0.03,0.05,0.08,0.1], "
            "subsample=[0.7,0.8,0.9,1.0], colsample_bytree=[0.7,0.8,0.9,1.0], "
            "min_child_weight=[1,3,5], reg_lambda=[0.5,1,2,5], "
            "scale_pos_weight=[0.25,0.5,1.0]. Imputer is fit within each fold. "
            f"p_star={p_star:.17g} was selected using train OOF probabilities to maximize macro-F1; "
            "test was not used to select the threshold. "
            f"Best-candidate CV RMSE={cv_result['rmse_mean']:.17g} (population sd={cv_result['rmse_std']:.17g}); "
            f"CV ROC-AUC={cv_result['auc_mean']:.17g} (population sd={cv_result['auc_std']:.17g}). "
            f"Exact test/train overlaps by all shared columns: {overlap_rows}. "
            "The supplied data files were used without dropping rows."
        ),
    }
    (OUT / "xgboost_lam.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Best params:", best_params)
    print("CV:", cv_result)
    print("p*:", p_star, "OOF macro-F1:", class_metrics(y.to_numpy(), oof_probability, p_star)["macro_f1"])
    print("Test probability metrics:", metrics_test)
    print("Test class metrics:", json.dumps(metrics_class))
    print("Baseline:", baseline)


if __name__ == "__main__":
    main()
