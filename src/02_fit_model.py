"""Step 2. Fit the model once by least squares (no inference here).

Output: results/regression_results.csv, results/year_specific_coefs.csv
"""

import numpy as np
import pandas as pd

from common import I_DELTA, I_THETA, L_STAR, RESULTS, Study, load_fall_days

RESULTS.mkdir(exist_ok=True)
d = load_fall_days()
s = Study(d)
X, y, n = s.X, s.y, len(d)
years = sorted(d.year.unique())

beta = np.linalg.lstsq(X, y, rcond=None)[0]
resid = y - X @ beta
r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
delta, theta = beta[I_DELTA], beta[I_THETA]
psi, kappa = beta[108], beta[111]

# Year-specific coefficients: b_y is the VIX jump per 1% fall at the year's average start.
b_y = beta[72:108]
pd.DataFrame({"year": years, "a_y": beta[:36], "g_y": beta[36:72], "b_y": b_y}).to_csv(
    RESULTS / "year_specific_coefs.csv", index=False
)
names = (
    [f"a_{t}" for t in years]
    + [f"g_{t}" for t in years]
    + [f"b_{t}" for t in years]
    + ["psi", "delta", "theta", "kappa"]
)
pd.DataFrame({"param": names, "estimate": beta}).to_csv(
    RESULTS / "regression_results.csv", index=False
)

# Fitted rise per 1% fall from the median previous close L*: mean over fall days of b_y + (delta + theta*P)*(L* - Lbar_y)
P, Lbar = d.P.values, d.Lbar_y.values
b_by_day = d.year.map(dict(zip(years, b_y))).values
J_star = (b_by_day + (delta + theta * P) * (L_STAR - Lbar)).mean()

print(f"N = {n:,}, parameters = {X.shape[1]}, rank = {np.linalg.matrix_rank(X)}, R2 = {r2:.4f}")
print(f"delta = {delta:.5f}   theta = {theta:.5f}   delta+theta = {delta + theta:.5f}")
print(f"psi = {psi:.6f}   kappa = {kappa:.4f}")
print(f"yearly jump b_y: min {b_y.min():.2f}, median {np.median(b_y):.2f}, max {b_y.max():.2f}")
print(f"fitted rise per 1% fall at L* = {L_STAR}, averaged over fall days: {J_star:.4f} VIX points")
