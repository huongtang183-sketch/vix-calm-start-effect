"""Step 4. Robustness: fixed list of alternative specifications, each run once.

Nothing was changed after seeing results. Wild cluster bootstrap-t, B = 2,999, seed 12345
re-seeded per row (delta and theta tested, null imposed); delta + theta: estimate and CR1 t.
Rows: drop psi; drop kappa; common gamma; add lambda*x*m(m-1); psi and kappa by period;
drop 2008/2009/2020; years from 2004; two-day non-overlapping changes (two start days);
plus 36 leave-one-year-out fits. Output: results/robustness.md
"""

import numpy as np, pandas as pd
from common import DATA, RESULTS

d0 = pd.read_csv(DATA / "clean" / "analysis_fall_days_vars.csv")


def build(d, terms=("psi", "kappa"), gcommon=False, lam=False, Pterms=False):
    yrs = sorted(d.year.unique())
    x = d.x.values
    m = d.m.values
    P = d.P.values
    x2 = x**2
    mm1 = m * (m - 1)
    Yd = np.stack([(d.year.values == t).astype(float) for t in yrs], 1)
    cols = [Yd, Yd * m[:, None]]
    cols.append(x[:, None] if gcommon else Yd * x[:, None])
    names = ["_"] * 0
    c = [x * m, P * x * m]  # delta, theta
    ex = []
    if "psi" in terms:
        ex.append(x2)
    if "kappa" in terms:
        ex.append(mm1)
    if lam:
        ex.append(x * mm1)
    if Pterms:
        ex += [P * x2, P * mm1]
    Z = np.column_stack(cols + c + ex)
    nb = Z.shape[1] - len(ex) - 2
    return Z, nb, nb + 1, nb + 2  # X, idx delta, idx theta, idx first extra


class Fit:
    def __init__(s, X, cl, G):
        s.X = X
        s.G = G
        s.cl = cl
        N = len(X)
        s.N = N
        s.A = np.linalg.pinv(X)
        k = np.linalg.matrix_rank(X)
        s.k = k
        s.c = G / (G - 1) * (N - 1) / (N - k)
        s.S = np.zeros((G, N))
        s.S[cl, np.arange(N)] = 1.0

    def tst(s, a, Yb):
        Bt = a @ (s.A @ Yb) if False else (a[:, None] * 0)


def cr1(F, a, Yb):  # a: row of A (vector); Yb N x b
    est = a @ Yb
    U = Yb - F.X @ (F.A @ Yb)
    V = F.c * ((F.S @ (a[:, None] * U)) ** 2).sum(0)
    return est, np.sqrt(V)


def wb(d, X, jd, jt, B=2999, seed=12345):
    yrs = sorted(d.year.unique())
    G = len(yrs)
    cl = d.year.map({t: i for i, t in enumerate(yrs)}).values
    y = d.dV.values
    F = Fit(X, cl, G)
    rng = np.random.default_rng(seed)
    out = {}
    for nm, j in (("delta", jd), ("theta", jt)):
        a = F.A[j]
        e, s = cr1(F, a, y[:, None])
        e = e[0]
        s = s[0]
        Xr = np.delete(X, j, 1)
        ur = y - Xr @ (np.linalg.pinv(Xr) @ y)
        f = y - ur
        ts = []
        done = 0
        while done < B:
            bc = min(250, B - done)
            W = (rng.integers(0, 2, size=(G, bc)) * 2 - 1).astype(float)[cl]
            Yb = f[:, None] + W * ur[:, None]
            b, sb = cr1(F, a, Yb)
            ts.append(b / sb)
            done += bc
        ts = np.concatenate(ts)
        out[nm] = (e, s, float(np.mean(np.abs(ts) >= abs(e / s))))
    a = F.A[jd] + F.A[jt]
    e, s = cr1(F, a, y[:, None])
    out["sum"] = (e[0], e[0] / s[0])
    return out


def point(d, X, jd, jt):
    yrs = sorted(d.year.unique())
    G = len(yrs)
    cl = d.year.map({t: i for i, t in enumerate(yrs)}).values
    y = d.dV.values
    F = Fit(X, cl, G)
    r = {}
    for nm, j in (("delta", jd), ("theta", jt)):
        e, s = cr1(F, F.A[j], y[:, None])
        r[nm] = (e[0], e[0] / s[0])
    return r


def fmt(v, n=5):
    return f"{v:.{n}f}"


def verdict(e, p, ref):
    return "same" if (np.sign(e) == np.sign(ref) and (p < 0.05) == False) else "changed"


L = []
Z, jd, jt, _ = build(d0)
M = wb(d0, Z, jd, jt)
RD, RT = M["delta"][0], M["theta"][0]
L.append(
    f'MAIN (B=2999 re-run): delta {RD:.5f} (CR1 SE {M["delta"][1]:.5f}, p {M["delta"][2]:.3f}); theta {RT:.5f} (SE {M["theta"][1]:.5f}, p {M["theta"][2]:.3f}); sum {M["sum"][0]:.5f} t {M["sum"][1]:.3f}'
)
rows = []
extra = []


def addrow(name, d, Z, jd, jt, note=""):
    o = wb(d, Z, jd, jt)
    rows.append(
        (
            name,
            len(d),
            o,
            verdict(o["delta"][0], o["delta"][2], RD),
            verdict(o["theta"][0], o["theta"][2], RT),
        )
    )
    return o


addrow("1 drop psi", d0, *build(d0, terms=("kappa",))[:3])
addrow("2 drop kappa", d0, *build(d0, terms=("psi",))[:3])
addrow("3 common gamma", d0, *build(d0, gcommon=True)[:3])
addrow("4 add lambda*x*mm1", d0, *build(d0, lam=True)[:3])
Z5, jd5, jt5, j0 = build(d0, Pterms=True)
o5 = addrow("5 P*x2, P*mm1", d0, Z5, jd5, jt5)
F5 = Fit(Z5, d0.year.map({t: i for i, t in enumerate(sorted(d0.year.unique()))}).values, 36)
b5 = F5.A @ d0.dV.values
extra.append(
    f"Row 5 extra coefficients: psi {b5[j0]:.6g}, kappa {b5[j0+1]:.6g}, P*x2 {b5[j0+2]:.6g}, P*mm1 {b5[j0+3]:.6g}"
)
# B
loo = []
for yv in sorted(d0.year.unique()):
    d = d0[d0.year != yv]
    r = point(d, *build(d)[:3])
    loo.append((yv, r["delta"][0], r["delta"][1], r["theta"][0], r["theta"][1]))
loo = pd.DataFrame(loo, columns=["year", "delta", "t_d", "theta", "t_t"])
loo["dchg"] = loo.delta - RD
loo["tchg"] = loo.theta - RT
# C
d = d0[~d0.year.isin([2008, 2009, 2020])]
addrow("6 drop 2008,2009,2020", d, *build(d)[:3])
d = d0[d0.year >= 2004]
addrow("7 years>=2004", d, *build(d)[:3])
raw = pd.read_csv(DATA / "sp500_vix_1990_2025.csv", parse_dates=["date"]).reset_index(drop=True)
an = pd.DataFrame(
    {
        "date": raw.date.values[2:],
        "S": raw.sp500_close.values[2:],
        "S2": raw.sp500_close.values[:-2],
        "V": raw.vix_close.values[2:],
        "L": raw.vix_close.values[:-2],
    }
)
an["m"] = -100 * np.log(an.S / an.S2)
an["dV"] = an.V - an.L
an["year"] = pd.to_datetime(an.date).dt.year
for off in (0, 1):
    t = an.iloc[off::2]
    t = t[t.m > 0].copy()  # fall = two-day log return<0 <=> m>0
    t["x"] = t.L - t.groupby("year").L.transform("mean")
    t["P"] = (t.year >= 2009).astype(int)
    t = t.reset_index(drop=True)
    cnt = t.groupby("year").size()
    addrow(f'8{"ab"[off]} 2-day nonoverlap start row {off} (n={len(t)})', t, *build(t)[:3])
# output
md = [
    "# Sensitivity checks: spec and results",
    "",
    "Code: src/04_robustness.py. Design builder as src/common.py: year dummies, year dummies*m, year dummies*x (g_y), then delta (x*m), theta (P*x*m), then psi (x^2), kappa (m(m-1)); P=1 for year>=2009. Clusters = calendar year. CR1 c=G/(G-1)*(N-1)/(N-k), k=rank (pinv used, matters only for 2-day rows with sparse years). Wild cluster bootstrap-t, Rademacher by year, null imposed (restricted fit drops the tested column), B=2999, seed 12345 re-seeded per row, draw order T1 (delta) then T2 (theta), two-sided p=share(|t*|>=|t|), 5% level. delta+theta: estimate and CR1 t only (contrast row a_delta+a_theta).",
    'Main model reference for verdict: delta -0.00605 (p 0.130, not rejected), theta -0.00653 (p 0.615, not rejected). Verdict = "same" iff sign equals main sign and not rejected at 5% (main status), else "changed". Change columns are relative to the point estimates of the main model re-fitted here.',
    "",
    "Definitions: (1) remove x^2; (2) remove m(m-1); (3) replace Yd*x by one common x column; (4) add lambda*x*m(m-1); (5) add P*x^2 and P*m(m-1); (6) drop years 2008,2009,2020; (7) years>=2004; (8) analysis rows from raw file rows 2..end: S_t,S_(t-2),V_t,V_(t-2); take rows 0,2,4,... (8a) or 1,3,5,... (8b) of these; m=-100 ln(S_t/S_(t-2)); L=V_(t-2); dV=V_t-V_(t-2); fall day = m>0 (two-day return<0); year = year of end date; x = L minus year mean L over that year's retained fall days; same terms. No gap-flag filtering applied in row 8 (raw consecutive rows used as is; weekends/holidays not treated specially).",
    "",
    L[0],
    "",
    "## A and C: bootstrap results",
    "",
    "| row | N | delta | SE | p | d-chg | theta | SE | p | t-chg | d+t | t(d+t) | verd d | verd t |",
    "|" + "---|" * 14,
]
for n, N, o, vd, vt in rows:
    md.append(
        f'| {n} | {N} | {o["delta"][0]:.5f} | {o["delta"][1]:.5f} | {o["delta"][2]:.3f} | {o["delta"][0]-RD:+.5f} | {o["theta"][0]:.5f} | {o["theta"][1]:.5f} | {o["theta"][2]:.3f} | {o["theta"][0]-RT:+.5f} | {o["sum"][0]:.5f} | {o["sum"][1]:.2f} | {vd} | {vt} |'
    )
md += (
    [""]
    + extra
    + [
        "",
        "## B: leave-one-year-out (36 fits, point estimates, CR1 t, no bootstrap)",
        "",
        f"delta_hat min {loo.delta.min():.5f} (drop {int(loo.year[loo.delta.idxmin()])}), max {loo.delta.max():.5f} (drop {int(loo.year[loo.delta.idxmax()])}); theta_hat min {loo.theta.min():.5f} (drop {int(loo.year[loo.theta.idxmin()])}), max {loo.theta.max():.5f} (drop {int(loo.year[loo.theta.idxmax()])})",
        "",
        "Five largest |change| in delta:",
        "",
        "| year dropped | delta | change | CR1 t |",
        "|---|---|---|---|",
    ]
)
for _, r in loo.reindex(loo.dchg.abs().sort_values(ascending=False).index).head(5).iterrows():
    md.append(f"| {int(r.year)} | {r.delta:.5f} | {r.dchg:+.5f} | {r.t_d:.2f} |")
md += [
    "",
    "Five largest |change| in theta:",
    "",
    "| year dropped | theta | change | CR1 t |",
    "|---|---|---|---|",
]
for _, r in loo.reindex(loo.tchg.abs().sort_values(ascending=False).index).head(5).iterrows():
    md.append(f"| {int(r.year)} | {r.theta:.5f} | {r.tchg:+.5f} | {r.t_t:.2f} |")
md += [
    "",
    "Full leave-one-out table:",
    "",
    "| year | delta | t | theta | t |",
    "|---|---|---|---|---|",
] + [
    f"| {int(r.year)} | {r.delta:.5f} | {r.t_d:.2f} | {r.theta:.5f} | {r.t_t:.2f} |"
    for _, r in loo.iterrows()
]
open(RESULTS / "robustness.md", "w").write("\n".join(md))
print("\n".join(md[:30]))
