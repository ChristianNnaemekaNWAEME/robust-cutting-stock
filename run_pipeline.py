"""
run_pipeline.py

End-to-end demonstration: simulated, correlated/contaminated production
order data -> robust demand prediction (SLTS+KLE hybrid, from the companion
slts-kle-hybrid project) -> cutting-stock plan (First-Fit-Decreasing) ->
material waste. Compared against a non-robust (OLS-based) prediction
pipeline and against the "oracle" plan built from the true piece lengths, to
measure how prediction robustness translates into actual material waste --
the quantity the proposed endeavor is centered on reducing.

Run with:  python3 src/run_pipeline.py
Outputs:   results/waste_comparison.csv
           results/waste_comparison.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from demand_model import simulate_production_orders, fit_robust_demand_model, fit_ridge_baseline
from cutting_stock import plan_summary

STOCK_LENGTH = 600.0


def evaluate_once(seed, contamination):
    X, y_true, beta_true = simulate_production_orders(
        contamination=contamination, random_state=seed
    )
    n = X.shape[0]
    split = int(n * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_true[:split], y_true[split:]

    # Robust (SLTS + Kibria-Lukman) pipeline. The hybrid estimator has no
    # intercept term, so the target must be centered before fitting (the
    # trimmed-sample mean is used for centering, since that is the
    # robust-consistent estimate of location once outliers are screened
    # out) and the same center added back at prediction time.
    beta_robust, inlier_mask, k_used = fit_robust_demand_model(
        X_train, y_train - np.median(y_train)
    )
    y_center = float(np.mean(y_train[inlier_mask]))
    y_pred_robust = X_test @ beta_robust + y_center

    # Naive OLS pipeline (no robustness to contamination, no regularization)
    beta_ols, *_ = np.linalg.lstsq(X_train, y_train - y_train.mean(), rcond=None)
    y_pred_ols = X_test @ beta_ols + y_train.mean()

    # Non-robust but regularized baseline (plain Ridge, no outlier screening)
    beta_ridge, intercept_ridge = fit_ridge_baseline(X_train, y_train)
    y_pred_ridge = X_test @ beta_ridge + intercept_ridge

    y_pred_robust = np.clip(y_pred_robust, 20, None)
    y_pred_ols = np.clip(y_pred_ols, 20, None)
    y_pred_ridge = np.clip(y_pred_ridge, 20, None)

    oracle_plan = plan_summary(y_test, STOCK_LENGTH)
    robust_plan = plan_summary(y_pred_robust, STOCK_LENGTH)
    ols_plan = plan_summary(y_pred_ols, STOCK_LENGTH)
    ridge_plan = plan_summary(y_pred_ridge, STOCK_LENGTH)

    # The metric that actually matters operationally: if you cut to the
    # *predicted* lengths but the *true* order lengths are what the
    # customer actually needs, how much real material is wasted (either
    # as true excess scrap, or as an under-cut piece that must be redone)?
    def realized_waste(y_pred):
        shortfall = np.maximum(y_test - y_pred, 0)   # piece too short: must be re-cut from fresh stock
        overage = np.maximum(y_pred - y_test, 0)      # piece too long: excess trimmed off as scrap
        rework_stock = shortfall.astype(bool).sum() * STOCK_LENGTH
        return float(np.sum(overage) + rework_stock)

    return {
        "seed": seed,
        "contamination": contamination,
        "oracle_waste_pct": 100 * oracle_plan["total_waste"] / oracle_plan["total_stock_used"],
        "robust_realized_waste": realized_waste(y_pred_robust),
        "ols_realized_waste": realized_waste(y_pred_ols),
        "ridge_realized_waste": realized_waste(y_pred_ridge),
        "robust_rmse": float(np.sqrt(np.mean((y_pred_robust - y_test) ** 2))),
        "ols_rmse": float(np.sqrt(np.mean((y_pred_ols - y_test) ** 2))),
        "ridge_rmse": float(np.sqrt(np.mean((y_pred_ridge - y_test) ** 2))),
    }


def main():
    contamination_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    n_repeats = 8
    rows = []
    for c in contamination_levels:
        for rep in range(n_repeats):
            seed = 2000 * int(c * 100) + rep
            rows.append(evaluate_once(seed, c))
            print(f"contamination={c:.2f} rep={rep} done")

    df = pd.DataFrame(rows)
    df.to_csv("results/waste_raw.csv", index=False)

    summary = df.groupby("contamination")[
        ["robust_realized_waste", "ols_realized_waste", "ridge_realized_waste",
         "robust_rmse", "ols_rmse", "ridge_rmse"]
    ].mean()
    summary.to_csv("results/waste_comparison.csv")
    print("\n=== Mean realized material waste (length units) by contamination ===")
    print(summary[["robust_realized_waste", "ols_realized_waste", "ridge_realized_waste"]].round(1))
    print("\n=== Mean prediction RMSE by contamination ===")
    print(summary[["robust_rmse", "ols_rmse", "ridge_rmse"]].round(3))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for col, label in [("robust_realized_waste", "SLTS+KLE (robust)"),
                        ("ols_realized_waste", "Naive OLS"),
                        ("ridge_realized_waste", "Plain Ridge")]:
        axes[0].plot(summary.index, summary[col], marker="o", label=label)
    axes[0].set_xlabel("Outlier contamination fraction")
    axes[0].set_ylabel("Mean realized material waste (length units)")
    axes[0].set_title("Material waste from cutting to predicted lengths")
    axes[0].legend()

    for col, label in [("robust_rmse", "SLTS+KLE (robust)"),
                        ("ols_rmse", "Naive OLS"),
                        ("ridge_rmse", "Plain Ridge")]:
        axes[1].plot(summary.index, summary[col], marker="o", label=label)
    axes[1].set_xlabel("Outlier contamination fraction")
    axes[1].set_ylabel("Prediction RMSE")
    axes[1].set_title("Upstream demand-prediction error")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("results/waste_comparison.png", dpi=150)
    print("\nSaved results/waste_comparison.png and .csv")


if __name__ == "__main__":
    main()
