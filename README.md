# Does the VIX Jump More When the S&P 500 Falls From a Calm Start?

On the 4,193 days from 1990 to 2025 on which the S&P 500 closed below the previous day's close, the rise in the VIX for a 1% fall does not detectably depend on the previous day's VIX close, in 1990–2008 or in 2009–2025, and the two periods do not detectably differ. A risk manager revaluing an option book can therefore apply a fixed number of VIX points per 1% fall. From the median previous close (17.75) the fitted rise is 1.36 points.

## Results

| Test | Estimate | p-value | Decision at 5% |
|---|---|---|---|
| Q1, 1990–2008 (δ) | −0.00605 | 0.130 | H₀₁ not rejected |
| Q1, 2009–2025 (δ + θ) | −0.01258 | 0.339 | H₀₁ not rejected |
| Q2 (θ) | −0.00653 | 0.615 | H₀₂ not rejected |

Estimates are in VIX points of rise per 1% fall, per +1 VIX point of the previous close. Method, variables and interpretation are in `report/vix_calm_start_report.pdf`.

## Reproduce

```
pip install -r requirements.txt
cd src
python 01_build_dataset.py
python 02_fit_model.py
python 03_inference.py
python 04_robustness.py
```

| Script | Output |
|---|---|
| `01_build_dataset.py` | `data/clean/` fall-day file |
| `02_fit_model.py` | `results/regression_results.csv`, `results/year_specific_coefs.csv` |
| `03_inference.py` | `results/inference.json` (estimates, p-values, intervals) |
| `04_robustness.py` | `results/robustness.md` (nine alternative versions of the model) |

In `results/robustness.md`, p-values below 0.05 mark a rejection: 17 of the 18 tests of δ and θ do not reject, as in the main model. The `changed` labels in its last two columns also flag a change of sign, so they are stricter than the decision.

## Data

See `data/README.md`. The S&P 500 closes before 2016-10-21 come from a GitHub mirror of Yahoo Finance data that is not checked against an original provider; on the 546 days it overlaps FRED, the two agree to within 0.04 index points.
