"""
demand_model.py

Simulates correlated, noisy order/material-usage data for a textile/sheet-
metal style cutting-stock operation, and fits a robust regression model
(reusing the SLTS + Kibria-Lukman hybrid from the companion slts-kle-hybrid
project) to predict required piece lengths from process/sensor features that
are themselves noisy and collinear -- a stand-in for the kind of production
telemetry (machine settings, material-batch properties, operator/shift
indicators) that in practice drives material requirements.

This is a simulation, clearly labeled as such, used to demonstrate the
end-to-end pipeline: robust statistical prediction feeding directly into a
cutting-stock solver. It is not a claim about any specific company's real
production data.
"""

import numpy as np
import sys
import os

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "slts-kle-hybrid", "src"),
)
from estimators import sparse_trimmed_least_squares, slts_kle_from_mask, fit_ols  # noqa: E402
from sklearn.linear_model import RidgeCV


def simulate_production_orders(n_orders=90, n_features=20, n_factors=4,
                                noise_sd=0.4, contamination=0.08,
                                base_length=180.0, random_state=0):
    """
    Simulate n_orders production orders. Each order has n_features
    correlated process measurements (batch width, roll tension, humidity,
    operator shift dummy variables, machine-speed readings, etc., all
    standardized), generated from a low-rank factor model to induce
    realistic multicollinearity among sensor/process readings. The true
    required piece length is a dense linear function of these features
    (reflecting that many process conditions jointly affect how much
    material a job actually consumes), plus noise, plus a contamination
    fraction of misrecorded/anomalous orders (e.g. scanner misreads,
    rework jobs).
    """
    rng = np.random.RandomState(random_state)
    n, p = n_orders, n_features

    factors = rng.normal(size=(n, n_factors))
    loadings = rng.normal(size=(n_factors, p))
    X = factors @ loadings + 0.3 * rng.normal(size=(n, p))
    X = (X - X.mean(0)) / X.std(0)

    beta_true = rng.uniform(1.0, 4.0, size=p) * rng.choice(
        [-1, 1], size=p, p=[0.3, 0.7]
    )
    lengths = base_length + X @ beta_true + noise_sd * base_length * 0.02 * rng.normal(size=n)

    n_out = int(round(contamination * n))
    if n_out > 0:
        out_idx = rng.choice(n, size=n_out, replace=False)
        lengths[out_idx] += rng.choice([-1, 1], size=n_out) * rng.uniform(60, 120, size=n_out)

    lengths = np.clip(lengths, 20, None)
    return X, lengths, beta_true


def fit_robust_demand_model(X_train, y_train):
    """Fit the SLTS + Kibria-Lukman hybrid from the companion project."""
    _, inlier_mask = sparse_trimmed_least_squares(X_train, y_train, random_state=0)
    beta_kl, k_used = slts_kle_from_mask(X_train, y_train, inlier_mask)
    return beta_kl, inlier_mask, k_used


def fit_ridge_baseline(X_train, y_train):
    model = RidgeCV(alphas=np.logspace(-2, 3, 50), cv=5).fit(X_train, y_train)
    return model.coef_, model.intercept_
