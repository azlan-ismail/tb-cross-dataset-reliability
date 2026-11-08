# Figures, tables and analyses for the article

These scripts regenerate every figure, table and secondary analysis in

> **High AUROC, Unreliable Thresholds: Uncertainty-Aware Threshold Calibration for Tuberculosis Chest-Radiograph Classifiers**
> (submitted to *Computerized Medical Imaging and Graphics*)

They read only saved per-image scores and estimates in `evidence/`. No GPU, data download or model fitting is needed.

## One command

From the repository root, with Python 3.10 or later:

```sh
python -m pip install -r analysis/requirements.txt
python analysis/reproduce.py            # outputs to analysis-output/
```

`reproduce.py` runs every script below in order. It then checks that the regenerated analysis files and table bodies are byte-identical to the copies saved in `evidence/analysis/`. It also confirms that the source files behind the plotted estimates are present and unchanged.

## Where each article item comes from

| Article item | Script | Saved output in `evidence/analysis/` |
|---|---|---|
| Fig. 1 Study design | `figure1_study_design.py` | (diagram only) |
| Fig. 2 External performance at transferred thresholds | `figure2_external_performance.py` | plots `external_performance_estimates.csv` |
| Table 2 External performance | `tables.py` | `tables/table2_external_performance.tex` |
| Sections 3.6, 4.2: threshold-selection uncertainty and local reference thresholds | `threshold_uncertainty.py` | `threshold_uncertainty.json` |
| Fig. 3 Threshold uncertainty and local-threshold specificity | `figures_3_5_S1.py` | `local_threshold_subgroups.json` |
| Section 3.7 and Algorithm 1: proposed uncertainty-aware threshold calibration; Section 4.3 | `threshold_calibration.py` | `threshold_calibration.json` |
| Table 3 Conventional versus uncertainty-aware rule | `tables.py` | `tables/table3_threshold_rules.tex` |
| Fig. 4 Local calibration by number of local TB positives | `figure4_local_calibration.py` | (figure from `threshold_calibration.json`) |
| Table 4 TBX11K referral rate and negative-subgroup specificity | `tables.py` | `tables/table4_negative_subgroups.tex` |
| Fig. 5 Score distributions and reliability | `figures_3_5_S1.py` | (figure from saved scores) |
| Supplementary Fig. S1 Matched encoder comparison | `figures_3_5_S1.py` | plots `encoder_comparison_estimates.csv` |
| Supplementary Table S4 Threshold-selection uncertainty | `tables.py` | `tables/tableS4_threshold_uncertainty.tex` |
| Supplementary Table S5 Probability error and calibration | `tables.py` | `tables/tableS5_calibration.tex` |

Supplementary Tables S2–S3 and S6–S7 report saved estimates from the experiment folders: `evidence/matched_resolution/`, `evidence/comparators/` and `evidence/preprocessing/`.

## Settings

- **Threshold-selection uncertainty:** 10,000 class-stratified bootstrap resamples of the Shenzhen validation set and each external cohort, seed 20261106. The threshold is re-selected in every resample.
- **Uncertainty-aware threshold:** Neyman–Pearson order-statistic rule (Tong, Feng and Li, 2018) with target miss rate α = 0.10 and confidence 1 − δ = 0.95, so at least 29 confirmed TB-positive images are needed. Local calibration uses 2,000 random draws per calibration size, seed 20261108.
- **Wilson intervals** are used for subgroup specificity.

## Notes on the saved estimate files

`external_performance_estimates.csv` and `encoder_comparison_estimates.csv` were prepared for an earlier draft. They are kept byte-for-byte, so their `figure` and `panel` columns use that draft's numbering:

| Draft label | Current item |
|---|---|
| `figure` 2, panels A–C | Fig. 2 |
| `figure` 3, panels A–B | Fig. S1 |
| `figure` 3, panel C | Table S7 |

`estimates_provenance.json` records the SHA-256 of each source file the estimates were taken from. Every one of those files is present, unchanged, under `evidence/`.
