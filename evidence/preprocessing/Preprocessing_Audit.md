# DINOv2 preprocessing diagnostic

Post hoc diagnostic, with source-only fitting and threshold selection. All conditions feed 224x224 tensors. HF denotes the saved checkpoint processor; TV denotes torchvision.

| Condition | Cohort | AUROC | Sensitivity | Specificity |
|---|---|---|---|---|
| hf256_crop224 | montgomery | 0.795043 | 0.465517 | 0.912500 |
| hf256_crop224 | tbx11k | 0.709439 | 0.675000 | 0.649310 |
| hf224_crop224 | montgomery | 0.819828 | 0.775862 | 0.625000 |
| hf224_crop224 | tbx11k | 0.598761 | 0.770000 | 0.308030 |
| tv256_crop224 | montgomery | 0.788793 | 0.534483 | 0.875000 |
| tv256_crop224 | tbx11k | 0.698912 | 0.700000 | 0.587202 |

## Replication diagnostics

{
  "pixel_max_abs_hf256_vs_tv256": 4.061624526977539,
  "features": {
    "hf256_crop224": {
      "reference": "original features",
      "feature_max_abs_difference": 2.9802322387695312e-05,
      "feature_mean_abs_difference": 1.5478698287552106e-06
    },
    "hf224_crop224": {
      "reference": "matched224 features",
      "feature_max_abs_difference": 0.6453964710235596,
      "feature_mean_abs_difference": 0.00359791936352849
    },
    "tv256_crop224": {
      "reference": "HF256 current features",
      "feature_max_abs_difference": 0.6355204582214355,
      "feature_mean_abs_difference": 0.006430561188608408
    }
  }
}

Compare HF256 versus HF224 to assess resize/crop geometry under one processor and inference implementation. Compare HF256 versus TV256 to assess implementation differences. Compare regenerated features with prior caches to detect additional numerical/runtime differences. Do not attribute the whole earlier score gap to cropping unless replication agrees. No target-selected replacement model is declared.