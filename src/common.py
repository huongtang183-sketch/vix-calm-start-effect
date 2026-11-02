"""Shared helpers: paths, data loading, regression design and wild cluster bootstrap.

Model (least squares, fall days only; y = calendar year, P = 1 from 2009):

    dV = a_y + g_y*x + psi*x^2 + [b_y + delta*x + theta*P*x]*m + kappa*m*(m-1) + u

dV: VIX change (points); m: S&P 500 fall size (%); x: previous-day VIX close
minus the year's mean of it on fall days. a_y, g_y, b_y are year-specific;
psi, delta, theta, kappa are common. Design columns 0-35 a_y, 36-71 g_y,
72-107 b_y, then 108 psi, 109 delta, 110 theta, 111 kappa.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
L_STAR = 17.750  # median previous-day VIX close on fall days (fixed reference start)
I_DELTA, I_THETA = 109, 110  # column positions of delta and theta in the design
SEED = 12345
B_MAIN = 9999


def load_fall_days() -> pd.DataFrame:
    return pd.read_csv(DATA / "clean" / "analysis_fall_days_vars.csv")


class Design:
    """Design matrix X with the pieces needed for fast CR1 statistics."""

    def __init__(self, X: np.ndarray, n_clusters: int):
        n, self.k = X.shape
        self.X = X
        self.A = np.linalg.solve(X.T @ X, X.T)  # (X'X)^-1 X'
        self.c = n_clusters / (n_clusters - 1) * (n - 1) / (n - self.k)  # CR1 factor


class Study:
    """Fall-day data, clusters (calendar years) and the designs used in the tests."""

    def __init__(self, d: pd.DataFrame):
        self.d = d
        years = sorted(d.year.unique())
        self.G, self.N = len(years), len(d)
        self.cl = d.year.map({t: i for i, t in enumerate(years)}).values
        self.y = d.dV.values
        x, m, P = d.x.values, d.m.values, d.P.values
        dummies = np.stack([(d.year.values == t).astype(float) for t in years], 1)
        self.X = np.hstack(
            [
                dummies,
                dummies * x[:, None],
                dummies * m[:, None],
                d[["x2", "xm"]].values,
                (P * d.xm.values)[:, None],
                d[["mm1"]].values,
            ]
        )
        # Alternative parametrisation: delta (earlier) and phi = delta + theta (later)
        X3 = self.X.copy()
        X3[:, I_DELTA], X3[:, I_THETA] = (1 - P) * d.xm.values, P * d.xm.values
        self.S = np.zeros((self.G, self.N))  # cluster-sum operator
        self.S[self.cl, np.arange(self.N)] = 1.0
        mk = lambda X: Design(X, self.G)
        self.full, self.full_phi = mk(self.X), mk(X3)
        self.no_delta = mk(np.delete(self.X, I_DELTA, 1))
        self.no_theta = mk(np.delete(self.X, I_THETA, 1))
        self.no_phi = mk(np.delete(X3, I_THETA, 1))

    def cr1(self, D: Design, Yb: np.ndarray, j: int):
        """Coefficient j and its CR1 cluster-robust SE, for each column of Yb."""
        beta = D.A @ Yb
        resid = Yb - D.X @ beta
        var = D.c * ((self.S @ (D.A[j][:, None] * resid)) ** 2).sum(0)
        return beta[j], np.sqrt(var)

    def wild_bootstrap_t(self, D_full, D_null, j, B, rng, impose_null, chunk=250):
        """Wild cluster bootstrap-t (Cameron, Gelbach and Miller, 2008).

        Rademacher weights, one per calendar year. With impose_null the draws come
        from the fit without the tested term (for p-values); otherwise from the full
        fit (for intervals). Returns the estimate, its CR1 SE and the bootstrap t's.
        """
        y = self.y
        est, se = (v[0] for v in self.cr1(D_full, y[:, None], j))
        if impose_null:
            resid = y - D_null.X @ (D_null.A @ y)
        else:
            resid = y - D_full.X @ (D_full.A @ y)
        fitted = y - resid
        t_star, done = [], 0
        while done < B:
            b = min(chunk, B - done)
            w = (rng.integers(0, 2, size=(self.G, b)) * 2 - 1).astype(float)[self.cl]
            bs, ss = self.cr1(D_full, fitted[:, None] + w * resid[:, None], j)
            t_star.append(bs / ss if impose_null else (bs - est) / ss)
            done += b
        return est, se, np.concatenate(t_star)
