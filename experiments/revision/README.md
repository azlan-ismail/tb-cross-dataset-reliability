# Revision analyses

Scripts for the manuscript revision. They read only the saved per-image scores in `evidence/` and need no GPU, data download or model fitting.

| Script | Output |
|---|---|
| `analysis.py REPO OUT` | `tier1_analysis.json`: threshold-selection uncertainty (validation re-selection propagated to external sensitivity), local reference thresholds, referral rate, ECE and calibration slope; 10,000 class-stratified resamples, seed 20261106 |
| `figs.py REPO JSON OUT` | Figures 3-5 (`Fig3`-`Fig5`, PDF and PNG) and `fig3c_subgroups.json` (Wilson subgroup specificity at local thresholds) |
| `tables.py JSON SUBGROUPS REPO` | LaTeX rows for the threshold, negative-subgroup and calibration tables |
| `fig_design.py OUT` | Figure 1, the study-design diagram |

Requires Python 3.10+, NumPy and Matplotlib. Saved outputs used in the manuscript are in `evidence/revision/`.
