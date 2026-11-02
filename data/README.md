# Data

`sp500_vix_1990_2025.csv`: date, S&P 500 close (`sp500_close`), VIX close (`vix_close`); 9,063 rows, 1990-01-02 to 2025-12-31.

| Series | Source | Note |
|---|---|---|
| VIX | Cboe `VIX_History.csv`, close | Official Cboe file. |
| S&P 500 to 2016-10-20 | GitHub mirror of Yahoo Finance `^GSPC` (vijinho/sp500) | Personal mirror; licence not stated; not checked against an original provider. |
| S&P 500 from 2016-10-21 | FRED `SP500` | On 546 overlap days the two sources differ by at most 0.04 index points. |

Merged on dates present in both series; 32 dates with a close in only one series are dropped (listed in `src/01_build_dataset.py`). The script that merged the raw files is not part of this repository. No imputation or interpolation.

`clean/` is written by `src/01_build_dataset.py`.
