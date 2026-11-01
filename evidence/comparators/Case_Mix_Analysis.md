# Post hoc analysis: negative-case composition

24 September 2026. Same 200 TB images, original fixed thresholds, no model fitting. Five thousand paired stratified bootstrap resamples preserve TB/healthy/sick group sizes and share resampled TB cases across comparisons (seed 20260925). Intervals are image-level, not verified patient-level. This analysis was specified after seeing the original results.

| Model | TB vs healthy AUROC (95% CI) | TB vs other illness AUROC (95% CI) | Difference (95% CI) |
|---|---|---|---|
| linear | 0.954 (0.935–0.971) | 0.850 (0.816–0.882) | 0.104 (0.083–0.127) |
| head_seed42 | 0.963 (0.943–0.979) | 0.894 (0.864–0.922) | 0.069 (0.052–0.087) |
| head_seed43 | 0.968 (0.950–0.983) | 0.892 (0.861–0.921) | 0.077 (0.058–0.098) |
| head_seed44 | 0.965 (0.945–0.981) | 0.891 (0.860–0.919) | 0.074 (0.056–0.094) |

The same positives and model scores are used throughout: this isolates the effect of which negative images enter the evaluation, but does not prove the pathological cause of errors. The mixed-cohort AUROC is a negative-count-weighted average of the two subgroup AUROCs. Sensitivity is unchanged because the TB images and thresholds are unchanged. Precision-recall metrics across these subsets would also reflect their differing TB fractions. These exploratory intervals are not multiplicity-adjusted.
