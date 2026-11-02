# Sensitivity checks: spec and results

Code: src/04_robustness.py. Design builder as src/common.py: year dummies, year dummies*m, year dummies*x (g_y), then delta (x*m), theta (P*x*m), then psi (x^2), kappa (m(m-1)); P=1 for year>=2009. Clusters = calendar year. CR1 c=G/(G-1)*(N-1)/(N-k), k=rank (pinv used, matters only for 2-day rows with sparse years). Wild cluster bootstrap-t, Rademacher by year, null imposed (restricted fit drops the tested column), B=2999, seed 12345 re-seeded per row, draw order T1 (delta) then T2 (theta), two-sided p=share(|t*|>=|t|), 5% level. delta+theta: estimate and CR1 t only (contrast row a_delta+a_theta).
Main model reference for verdict: delta -0.00605 (p 0.130, not rejected), theta -0.00653 (p 0.615, not rejected). Verdict = "same" iff sign equals main sign and not rejected at 5% (main status), else "changed". Change columns are relative to the point estimates of the main model re-fitted here.

Definitions: (1) remove x^2; (2) remove m(m-1); (3) replace Yd*x by one common x column; (4) add lambda*x*m(m-1); (5) add P*x^2 and P*m(m-1); (6) drop years 2008,2009,2020; (7) years>=2004; (8) analysis rows from raw file rows 2..end: S_t,S_(t-2),V_t,V_(t-2); take rows 0,2,4,... (8a) or 1,3,5,... (8b) of these; m=-100 ln(S_t/S_(t-2)); L=V_(t-2); dV=V_t-V_(t-2); fall day = m>0 (two-day return<0); year = year of end date; x = L minus year mean L over that year's retained fall days; same terms. No gap-flag filtering applied in row 8 (raw consecutive rows used as is; weekends/holidays not treated specially).

MAIN (B=2999 re-run): delta -0.00605 (CR1 SE 0.00349, p 0.124); theta -0.00653 (SE 0.00827, p 0.621); sum -0.01258 t -1.639

## A and C: bootstrap results

| row | N | delta | SE | p | d-chg | theta | SE | p | t-chg | d+t | t(d+t) | verd d | verd t |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 drop psi | 4193 | -0.00968 | 0.00509 | 0.037 | -0.00363 | -0.00950 | 0.01039 | 0.744 | -0.00297 | -0.01918 | -1.55 | changed | same |
| 2 drop kappa | 4193 | 0.00636 | 0.00381 | 0.298 | +0.01241 | 0.00478 | 0.00545 | 0.272 | +0.01131 | 0.01114 | 1.92 | changed | changed |
| 3 common gamma | 4193 | 0.00034 | 0.00434 | 0.934 | +0.00639 | -0.03109 | 0.00855 | 0.248 | -0.02456 | -0.03075 | -4.22 | changed | same |
| 4 add lambda*x*mm1 | 4193 | 0.00093 | 0.00571 | 0.869 | +0.00698 | -0.00560 | 0.00809 | 0.644 | +0.00093 | -0.00467 | -0.44 | changed | same |
| 5 P*x2, P*mm1 | 4193 | -0.00468 | 0.00282 | 0.140 | +0.00137 | -0.01182 | 0.00909 | 0.242 | -0.00530 | -0.01650 | -1.91 | same | same |
| 6 drop 2008,2009,2020 | 3847 | -0.02259 | 0.01482 | 0.228 | -0.01654 | 0.03825 | 0.02146 | 0.127 | +0.04478 | 0.01567 | 0.91 | same | changed |
| 7 years>=2004 | 2508 | -0.00425 | 0.00238 | 0.498 | +0.00180 | -0.01031 | 0.00717 | 0.544 | -0.00378 | -0.01455 | -2.03 | same | same |
| 8a 2-day nonoverlap start row 0 (n=2033) | 2033 | -0.00662 | 0.01071 | 0.549 | -0.00057 | -0.03462 | 0.01524 | 0.444 | -0.02809 | -0.04124 | -4.52 | same | same |
| 8b 2-day nonoverlap start row 1 (n=1987) | 1987 | -0.01080 | 0.00726 | 0.075 | -0.00475 | 0.00217 | 0.00885 | 0.805 | +0.00870 | -0.00863 | -0.73 | same | changed |

Row 5 extra coefficients: psi 0.000483175, kappa 0.0501494, P*x2 -0.00262319, P*mm1 0.0945757

## B: leave-one-year-out (36 fits, point estimates, CR1 t, no bootstrap)

delta_hat min -0.01783 (drop 2008), max -0.00475 (drop 1998); theta_hat min -0.01101 (drop 1998), max 0.01571 (drop 2020)

Five largest |change| in delta:

| year dropped | delta | change | CR1 t |
|---|---|---|---|
| 2008 | -0.01783 | -0.01178 | -1.20 |
| 2020 | -0.00891 | -0.00286 | -1.95 |
| 1998 | -0.00475 | +0.00130 | -1.66 |
| 2018 | -0.00483 | +0.00122 | -1.64 |
| 1990 | -0.00488 | +0.00117 | -1.67 |

Five largest |change| in theta:

| year dropped | theta | change | CR1 t |
|---|---|---|---|
| 2020 | 0.01571 | +0.02223 | 1.24 |
| 2008 | 0.00592 | +0.01244 | 0.32 |
| 1998 | -0.01101 | -0.00448 | -1.53 |
| 2010 | -0.00886 | -0.00234 | -1.22 |
| 2018 | -0.00483 | +0.00169 | -0.57 |

Full leave-one-out table:

| year | delta | t | theta | t |
|---|---|---|---|---|
| 1990 | -0.00488 | -1.67 | -0.00662 | -0.81 |
| 1991 | -0.00522 | -1.64 | -0.00608 | -0.74 |
| 1992 | -0.00603 | -1.73 | -0.00648 | -0.78 |
| 1993 | -0.00610 | -1.74 | -0.00653 | -0.79 |
| 1994 | -0.00604 | -1.72 | -0.00643 | -0.78 |
| 1995 | -0.00603 | -1.73 | -0.00652 | -0.79 |
| 1996 | -0.00611 | -1.74 | -0.00651 | -0.79 |
| 1997 | -0.00687 | -1.81 | -0.00666 | -0.79 |
| 1998 | -0.00475 | -1.66 | -0.01101 | -1.53 |
| 1999 | -0.00606 | -1.72 | -0.00634 | -0.76 |
| 2000 | -0.00589 | -1.74 | -0.00741 | -0.89 |
| 2001 | -0.00669 | -1.75 | -0.00582 | -0.69 |
| 2002 | -0.00609 | -1.65 | -0.00598 | -0.71 |
| 2003 | -0.00629 | -1.74 | -0.00633 | -0.76 |
| 2004 | -0.00621 | -1.75 | -0.00642 | -0.77 |
| 2005 | -0.00606 | -1.73 | -0.00645 | -0.78 |
| 2006 | -0.00620 | -1.74 | -0.00635 | -0.77 |
| 2007 | -0.00623 | -1.72 | -0.00664 | -0.80 |
| 2008 | -0.01783 | -1.20 | 0.00592 | 0.32 |
| 2009 | -0.00560 | -1.67 | -0.00613 | -0.67 |
| 2010 | -0.00664 | -1.86 | -0.00886 | -1.22 |
| 2011 | -0.00587 | -1.69 | -0.00790 | -1.00 |
| 2012 | -0.00616 | -1.75 | -0.00677 | -0.82 |
| 2013 | -0.00595 | -1.72 | -0.00636 | -0.77 |
| 2014 | -0.00601 | -1.73 | -0.00650 | -0.79 |
| 2015 | -0.00640 | -1.80 | -0.00747 | -0.93 |
| 2016 | -0.00569 | -1.68 | -0.00565 | -0.66 |
| 2017 | -0.00600 | -1.73 | -0.00647 | -0.78 |
| 2018 | -0.00483 | -1.64 | -0.00483 | -0.57 |
| 2019 | -0.00605 | -1.73 | -0.00642 | -0.77 |
| 2020 | -0.00891 | -1.95 | 0.01571 | 1.24 |
| 2021 | -0.00581 | -1.70 | -0.00655 | -0.80 |
| 2022 | -0.00621 | -1.75 | -0.00661 | -0.78 |
| 2023 | -0.00602 | -1.73 | -0.00648 | -0.78 |
| 2024 | -0.00614 | -1.76 | -0.00822 | -1.10 |
| 2025 | -0.00555 | -1.67 | -0.00750 | -0.96 |