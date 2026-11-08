# High AUROC, unreliable thresholds: tuberculosis chest-radiograph classifiers

Code and evaluation records for the article:

> Azlan Ismail, Muhamad Harith Hadi, Sofianita Mutalib, Mohd Suffian Sulaiman and Jafar AbuKhait. **High AUROC, Unreliable Thresholds: Uncertainty-Aware Threshold Calibration for Tuberculosis Chest-Radiograph Classifiers.** Submitted to *Computerized Medical Imaging and Graphics*.

Correspondence: azlanismail@uitm.edu.my.

This is a research archive. It is not a clinical diagnostic system. The experiments measure agreement with public-dataset labels. Clinical validity, patient independence and pretraining independence are not established.

## Study in brief

Classifiers built on frozen RAD-DINO features (one logistic regression and three seeded neural classifiers) were developed on a Shenzhen split: 529 training and 133 validation images, 69 of them TB-positive. The threshold reaching at least 90% validation sensitivity was transferred unchanged to two external cohorts:

- Montgomery: 138 images.
- A duplicate-screened TBX11K cohort: 1,794 images (200 TB, 800 healthy and 794 other illness).

The article shows three things:

1. **High AUROC did not guarantee a reliable operating point.** The validation-selected threshold was highly uncertain with 69 positives.
2. **A calibration method is proposed and tested.** An uncertainty-aware threshold calibration (a Neyman–Pearson order-statistic rule) needs at least 29 local TB-positive images. It met the 90% sensitivity target in most repeated calibrations, at a stated referral cost.
3. **Other-illness negatives drove the remaining over-referral.** Local thresholds did not remove it.

## Reproduce the article's figures and tables from saved scores (no GPU)

```sh
python -m pip install -r analysis/requirements.txt
python analysis/reproduce.py
```

This regenerates Figures 1–5, Supplementary Figure S1, Tables 2–4, Supplementary Tables S4–S5, and the threshold-uncertainty and calibration analyses. It then checks that the outputs match the saved copies in `evidence/analysis/`. See [analysis/README.md](analysis/README.md) for the script behind each figure and table.

## Verify the packaged records

These two checks use only the Python standard library:

```sh
python scripts/verify_artifacts.py
python scripts/reproduce_point_metrics.py
```

- `verify_artifacts.py` checks packaged file hashes, source syntax and the archived execution receipts.
- `reproduce_point_metrics.py` recalculates the point estimates from the saved per-image scores and compares them with the saved results.

## Repository map

| Folder | Contents | Article |
|---|---|---|
| `analysis/` | Scripts that regenerate every figure, table and secondary analysis from saved scores | All figures and Tables 2–4, S4–S5 |
| `evidence/analysis/` | Saved analysis outputs, table bodies and plotted estimates | As above |
| `experiments/development/` | Shenzhen split, RAD-DINO feature extraction, classifier fitting and threshold selection (Colab) | Sections 3.1–3.5 |
| `experiments/external/` | TBX11K duplicate screening, cohort lock and external evaluation | Sections 3.2–3.3 |
| `experiments/matched_resolution/` | Matched RAD-DINO and DINOv2 comparison at 224 and 518 pixels | Supplementary S2 |
| `experiments/comparators/` | Initial DINOv2 and DenseNet121 comparators and case-mix analysis | Supplementary S4 |
| `experiments/preprocessing/` | DINOv2 resize–crop diagnostic | Supplementary S4 |
| `evidence/` (other folders) | Manifests, per-image scores, estimates, logs and completion receipts for each experiment | — |
| `validation/` | Recomputed point metrics | — |
| `scripts/` | Portable integrity and point-metric checks | — |
| `docs/` | Reproduction guide and archive status | — |

The historical experiment scripts keep their original Colab and Drive paths so that their recorded execution hashes stay valid. Read [docs/REPRODUCTION.md](docs/REPRODUCTION.md) before re-running them.

`SOURCE_MAP.csv` records where each copied artifact came from, with its SHA-256 and any transformation. `ARTIFACTS.sha256` covers all packaged code and records.

Raw radiographs, dataset archives, trained model binaries and credentials are not included. Obtain the images from the Shenzhen and Montgomery collections (NLM Open-i) and from TBX11K (<https://github.com/yun-liu/Tuberculosis>). The related PulmoVision application is a separate project: <https://github.com/HarithHadi/PulmoVision>.

## Licence and citation

Code is released under the [MIT licence](LICENSE) and derived evaluation records under [CC BY 4.0](LICENSE-DATA.md); see [NOTICE.md](NOTICE.md). An archived release with a DOI will be deposited on Zenodo when the article is accepted. Until then, please cite the repository using [CITATION.cff](CITATION.cff).
