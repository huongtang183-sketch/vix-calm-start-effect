"""Step 1. Build the daily dataset and the fall-day analysis file.

Input : data/sp500_vix_1990_2025.csv  (date, S&P 500 close, VIX close; see README, Data)
Output: data/clean/analysis_all_days.csv, data/clean/analysis_fall_days_vars.csv
Rules : no imputation; a change is the pair of consecutive remaining rows; a fall
        day is a strictly negative S&P 500 log return.
"""

import numpy as np
import pandas as pd

from common import DATA

# Dates with a close in only one of the two series (dropped before merging).
ONE_SERIES_DATES = pd.to_datetime(
    [
        "1991-03-01",
        "1997-01-31",
        "1997-11-26",
        "1999-12-31",
        "2004-06-11",
        "2022-05-30",
        "2022-06-20",
        "2022-07-04",
        "2022-09-05",
        "2022-11-24",
        "2023-01-16",
        "2023-02-20",
        "2023-05-29",
        "2023-06-19",
        "2023-07-04",
        "2023-09-04",
        "2023-11-23",
        "2024-01-15",
        "2024-02-19",
        "2024-05-27",
        "2024-06-19",
        "2024-07-04",
        "2024-09-02",
        "2024-11-28",
        "2025-01-09",
        "2025-01-20",
        "2025-02-17",
        "2025-05-26",
        "2025-06-19",
        "2025-07-04",
        "2025-09-01",
        "2025-11-27",
    ]
)
SPLIT_YEAR = 2009  # P = 1 from this year (fixed before estimation, see README)

raw = pd.read_csv(DATA / "sp500_vix_1990_2025.csv", parse_dates=["date"])
raw = raw.sort_values("date").reset_index(drop=True)
assert raw.notna().all().all() and not raw.date.duplicated().any()

d = pd.DataFrame({"date": raw.date, "year": raw.date.dt.year, "V": raw.vix_close})
d["L"] = raw.vix_close.shift()  # previous-day VIX close (start level)
d["dV"] = d.V - d.L  # VIX change, points
d["ret"] = np.log(raw.sp500_close / raw.sp500_close.shift())
d["m"] = -100 * d.ret  # fall size, percent (positive on fall days)
d["fall"] = (d.ret < 0).astype(int)

# Flag (do not delete) changes that span a holiday/missing date.
prev = raw.date.shift()
d["gap_days"] = (raw.date - prev).dt.days
d["gap_flag"] = [
    0 if pd.isna(p) else int(g > 4 or ((ONE_SERIES_DATES > p) & (ONE_SERIES_DATES < c)).any())
    for p, g, c in zip(prev, d.gap_days, raw.date)
]
d = d.iloc[1:].reset_index(drop=True)  # first row has no previous close

falls = d[d.fall == 1].copy()
per_year = falls.groupby("year").size()
assert (per_year >= 4).all(), "a year has fewer than 4 fall days"

# Within-year start level and the model terms.
falls["Lbar_y"] = falls.groupby("year").L.transform("mean")
falls["x"] = falls.L - falls.Lbar_y
falls["x2"] = falls.x**2
falls["xm"] = falls.x * falls.m
falls["mm1"] = falls.m * (falls.m - 1)
falls["P"] = (falls.year >= SPLIT_YEAR).astype(int)

out = DATA / "clean"
out.mkdir(exist_ok=True)
d.to_csv(out / "analysis_all_days.csv", index=False, date_format="%Y-%m-%d")
falls.to_csv(out / "analysis_fall_days_vars.csv", index=False, date_format="%Y-%m-%d")

print(f"{raw.date.min().date()} to {raw.date.max().date()}: {len(d):,} daily changes")
print(
    f"fall days {len(falls):,} ({len(falls) / len(d):.1%}); per year {per_year.min()}-{per_year.max()}"
)
print(f"changes spanning a holiday or missing date: {int(d.gap_flag.sum())} (kept)")
