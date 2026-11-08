# Cross-dataset reliability of tuberculosis classification

Research code and evaluation evidence for exploratory cross-dataset tuberculosis classification, including preprocessing, threshold transfer and negative-case composition.

**Authors:** Azlan Ismail, Muhamad Harith Hadi, Sofianita Mutalib, Mohd Suffian Sulaiman and Jafar AbuKhait. Correspondence: azlanismail@uitm.edu.my. Harith: m.harith.hadi@gmail.com.

This repository contains experiment code, plotting resources and available saved evaluation evidence. It is not a clinical diagnostic system. The experiments assess agreement with public-dataset labels; clinical validity, patient independence and pretraining independence remain unestablished.

## Start here

- [Reproduction instructions](docs/REPRODUCTION.md) distinguish saved-score checks from GPU experiments.
- [Archive completeness and release status](docs/RELEASE_STATUS.md) lists the material still needed for a complete experiment archive.

## Scientific scope

The exploratory study develops classifiers using one Shenzhen split (529 training and 133 validation images) and evaluates Montgomery (138 images) and a duplicate-screened TBX11K validation-derived cohort (1,794 images: 200 TB-labelled, 800 healthy and 794 other-illness images). Development-selected thresholds are transferred unchanged.

The experiments examine discrimination and threshold transfer, specificity across negative groups, initial RAD-DINO/DINOv2/DenseNet121 pipeline comparisons, matched encoder–resolution conditions, and a post hoc DINOv2 preprocessing diagnostic. Later comparisons were designed after earlier external results were observed. They do not isolate a causal effect of medical pretraining or provide confirmatory clinical evidence.

## Verify saved results without a GPU

Use Python 3.10 or later, from the repository root. These two scripts use only the Python standard library and do not download data or train models.

```sh
python scripts/verify_artifacts.py
python scripts/reproduce_point_metrics.py
```

The first checks packaged file hashes, source syntax and archived execution receipts. The second independently recalculates point estimates from available per-image scores for the initial classifiers and comparators. It does **not** recalculate bootstrap intervals or recover missing per-image outputs for the matched-resolution and preprocessing experiments.

## Revision analyses from saved scores

The threshold-selection uncertainty, local reference threshold, referral-rate, calibration and uncertainty-aware threshold analyses reported in the manuscript are computed only from the saved per-image scores, without a GPU. They need NumPy and Matplotlib:

```sh
python experiments/revision/analysis.py . OUT          # OUT/tier1_analysis.json (10,000 resamples, seed 20261106)
python experiments/revision/figs.py . OUT/tier1_analysis.json OUT   # Figures 3, 5, S1 and fig3c_subgroups.json
python experiments/revision/tables.py OUT/tier1_analysis.json OUT/fig3c_subgroups.json .   # LaTeX table rows
python experiments/revision/fig_design.py OUT          # Figure 1 (study design)
python experiments/revision/remedy.py . OUT            # OUT/remedy_analysis.json (uncertainty-aware threshold rule)
python experiments/revision/remedy_outputs.py OUT/remedy_analysis.json . OUT   # Figure 4 and table rows
```

The outputs used in the manuscript are saved in `evidence/revision/`.

## Repository map

| Folder | Contents |
|---|---|
| `experiments/development/` | Corrected initial RAD-DINO study, preparation, training, evaluation and Colab notebooks |
| `experiments/external/` | TBX11K source audit and external evaluation scripts |
| `experiments/comparators/` | Initial DINOv2/DenseNet121 comparison and case-mix analysis |
| `experiments/matched_resolution/` | Matched RAD-DINO/DINOv2 conditions and paired analysis |
| `experiments/preprocessing/` | DINOv2 resize–crop diagnostic |
| `experiments/revision/` | Threshold-uncertainty, calibration and figure scripts run on saved scores |
| `evidence/` | Available manifests, scores, estimates, logs and completion receipts; `evidence/revision/` holds the revision analysis outputs |
| `figures/` | Plotting code and saved figure data; no model fitting |
| `scripts/` | Portable checks for this packaged archive |
| `docs/` | Reproduction and completeness notes |

Historical scripts retain their original Colab/Drive paths so that recorded execution hashes remain meaningful. They are not all standalone commands from this repository layout. Read the reproduction instructions before running them. Historical protocol status statements describe their date of creation; later completion receipts and saved results are under `evidence/`.

`SOURCE_MAP.csv` records each copied artifact's original project-relative location, original SHA-256, repository SHA-256 and any transformation. Some Windows Python files were normalized to LF; the two development scripts then match the recorded Colab execution hashes. Notebook outputs and execution counts were cleared. Evidence files were copied byte-for-byte. `ARTIFACTS.sha256` covers the packaged source and evidence.

Raw radiographs, dataset ZIPs, trained model binaries and credentials are not included. The related PulmoVision FYP application is a separate project: <https://github.com/HarithHadi/PulmoVision>.

## Licence and citation

Code is released under the [MIT licence](LICENSE) and derived evaluation records under [CC BY 4.0](LICENSE-DATA.md); see [NOTICE.md](NOTICE.md). Releases are archived on Zenodo. Please cite the archive using [CITATION.cff](CITATION.cff).
