# Matched-resolution experiment

Exploratory comparison: frozen encoders, identical preprocessing at each resolution, identical train-only scaling and balanced linear classifier. Thresholds selected only on Shenzhen validation. Montgomery n=138; locked TBX11K n=1794.

| Cohort | Model | AUROC (95% CI) | Sensitivity | Specificity |
|---|---|---|---|---|
| montgomery | rad_224 | 0.924 (0.878–0.961) | 0.810 | 0.863 |
| montgomery | dino_224 | 0.812 (0.734–0.880) | 0.776 | 0.650 |
| montgomery | rad_518 | 0.958 (0.925–0.981) | 0.810 | 0.887 |
| montgomery | dino_518 | 0.703 (0.614–0.787) | 0.638 | 0.625 |
| tbx11k | rad_224 | 0.865 (0.835–0.894) | 0.885 | 0.606 |
| tbx11k | dino_224 | 0.583 (0.540–0.626) | 0.765 | 0.309 |
| tbx11k | rad_518 | 0.902 (0.877–0.926) | 0.915 | 0.637 |
| tbx11k | dino_518 | 0.594 (0.551–0.636) | 0.585 | 0.529 |

## Paired AUROC differences

| Cohort | Contrast | Difference (95% CI) |
|---|---|---|
| montgomery | rad_224 minus dino_224 | 0.112 (0.035–0.194) |
| montgomery | rad_518 minus dino_518 | 0.255 (0.176–0.337) |
| montgomery | rad_518 minus rad_224 | 0.033 (0.005–0.067) |
| montgomery | dino_518 minus dino_224 | -0.109 (-0.211–-0.008) |
| montgomery | encoder gap at 518 minus encoder gap at 224 | 0.143 (0.037–0.252) |
| tbx11k | rad_224 minus dino_224 | 0.282 (0.237–0.327) |
| tbx11k | rad_518 minus dino_518 | 0.309 (0.262–0.355) |
| tbx11k | rad_518 minus rad_224 | 0.037 (0.023–0.052) |
| tbx11k | dino_518 minus dino_224 | 0.010 (-0.030–0.053) |
| tbx11k | encoder gap at 518 minus encoder gap at 224 | 0.027 (-0.018–0.071) |

## Negative case mix

| Model | Negative group | AUROC | Specificity |
|---|---|---|---|
| rad_224 | health | 0.939 | 0.854 |
| rad_224 | sick | 0.791 | 0.356 |
| dino_224 | health | 0.657 | 0.370 |
| dino_224 | sick | 0.509 | 0.247 |
| rad_518 | health | 0.954 | 0.844 |
| rad_518 | sick | 0.851 | 0.428 |
| dino_518 | health | 0.676 | 0.642 |
| dino_518 | sick | 0.511 | 0.414 |

Intervals use 10,000 paired unstratified image bootstrap draws (seed 20260925), conditional on fitted models and thresholds. They do not capture training-split uncertainty. Subgroup values are point estimates. Multiple comparisons are exploratory and unadjusted.

Identical input processing controls the resolution/transform difference in this experiment. It does not isolate a causal effect of medical pretraining: checkpoint training histories differ. Patient/source independence and pretraining exposure are unverified. No expert clinical review was performed. Earlier target outcomes informed the choice of this extension. No target data were used to fit classifiers or choose thresholds.