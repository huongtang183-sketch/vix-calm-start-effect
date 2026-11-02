"""Step 3. Tests and intervals: wild cluster bootstrap-t, clustered by calendar year.

Tests      : H0 delta = 0 (Q1), H0 theta = 0 (Q2), H0 delta + theta = 0 (later years).
             Null imposed, Rademacher weights by year, B = 9999, seed 12345, 5% level.
Intervals  : bootstrap-t for delta, theta, delta + theta.
Output     : results/inference.json
"""

import json

import numpy as np

from common import B_MAIN, I_DELTA, I_THETA, RESULTS, SEED, Study, load_fall_days

d = load_fall_days()
s = Study(d)
rng = np.random.default_rng(SEED)

# (name, full design, null design, column): phi = delta + theta is tested by reparametrisation
TESTS = [
    ("delta", s.full, s.no_delta, I_DELTA),
    ("theta", s.full, s.no_theta, I_THETA),
    ("delta+theta", s.full_phi, s.no_phi, I_THETA),
]
out = {}
for name, D, D0, j in TESTS:  # p-values: null imposed
    est, se, t_star = s.wild_bootstrap_t(D, D0, j, B_MAIN, rng, impose_null=True)
    out[name] = {
        "estimate": est,
        "cr1_se": se,
        "t": est / se,
        "p": float(np.mean(np.abs(t_star) >= abs(est / se))),
    }
for name, D, _, j in TESTS:  # intervals: null not imposed
    est, se, t_star = s.wild_bootstrap_t(D, None, j, B_MAIN, rng, impose_null=False)
    lo, hi = np.quantile(t_star, [0.025, 0.975])
    out[name]["interval_95"] = [est - hi * se, est - lo * se]

RESULTS.mkdir(exist_ok=True)
json.dump(out, open(RESULTS / "inference.json", "w"), indent=1)
for k in ("delta", "theta", "delta+theta"):
    r = out[k]
    print(
        f"{k:12s} est {r['estimate']:+.5f}  CR1 SE {r['cr1_se']:.5f}  p {r['p']:.3f}  "
        f"95% [{r['interval_95'][0]:+.5f}, {r['interval_95'][1]:+.5f}]"
    )
print("Size: interval x 18 = change in the rise between previous closes of 12 and 30")
for k in ("delta", "delta+theta"):
    lo, hi = out[k]["interval_95"]
    print(f"{k:12s} {18 * lo:+.2f} to {18 * hi:+.2f}")
