from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHAP_DIR = ROOT / "shap_figures"


def main():
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shap_path = SHAP_DIR / "case1_xgb_lam_shap_oof.csv.gz"
    values_path = SHAP_DIR / "case1_xgb_lam_shap_values_X.csv.gz"
    shap_values = pd.read_csv(shap_path)
    feature_values = pd.read_csv(values_path)
    assert shap_values.shape == feature_values.shape
    assert np.isfinite(shap_values.to_numpy()).all()
    assert np.isfinite(feature_values.to_numpy()).all()

    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(feature_values), min(30000, len(feature_values)), replace=False))
    S = shap_values.iloc[idx].to_numpy()
    X = feature_values.iloc[idx]

    shap.summary_plot(S, X, max_display=11, show=False)
    ax = plt.gca()
    ax.axvline(0, color="#555555", linewidth=.8)
    ax.set_xlabel("SHAP value (log-odds contribution)")
    plt.title("Case 1 XGBoost - 10-fold OOF SHAP summary")
    plt.savefig(SHAP_DIR / "case1_xgb_lam_shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    shap.summary_plot(S, X, plot_type="bar", max_display=11, show=False)
    plt.gca().set_xlabel("mean |SHAP| (log-odds contribution)")
    plt.title("Case 1 XGBoost - mean absolute OOF SHAP")
    plt.savefig(SHAP_DIR / "case1_xgb_lam_shap_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    shap.dependence_plot("view_count", S, X, interaction_index="add_to_detail_ratio", show=False)
    ax = plt.gca()
    ax.axhline(0, color="#555555", linewidth=.8)
    ax.set_xlim(float(X.view_count.min()), float(X.view_count.quantile(.99)))
    ax.set_xlabel("view_count")
    ax.set_ylabel("SHAP value (log-odds contribution)")
    plt.title("SHAP dependence: view_count (color: add_to_detail_ratio)")
    plt.savefig(SHAP_DIR / "case1_xgb_lam_shap_dependence_view_count.png", dpi=200, bbox_inches="tight")
    plt.close()

    shap_values.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").to_csv(
        SHAP_DIR / "case1_xgb_lam_shap_importance.csv", header=True
    )
    print("SHAP", shap.__version__, "rows", len(shap_values), "plot sample", len(idx))


if __name__ == "__main__":
    main()
