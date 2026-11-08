# Reproduction guide

## 1. Checks that work from this repository

Run from the repository root with Python 3.10 or later:

```sh
python scripts/verify_artifacts.py
python scripts/reproduce_point_metrics.py
```

The artifact check validates the packaged SHA-256 manifest, copied-artifact hashes, Python syntax, JSON/notebook readability and available files covered by saved execution receipts. It reports receipt-listed assets absent from this repository separately. A successful check establishes consistency of packaged bytes, not clinical validity or completeness of the original experiment archive.

The point-metric check uses saved labels, scores and frozen thresholds to calculate AUROC (with tied-score handling), average precision, sensitivity, specificity, Brier score and confusion counts. It compares eight initial classifier/cohort rows and sixteen initial-comparator rows with their saved estimates. The RAD-DINO rows occur in both analyses, so these are 24 verification records, not 24 independent experiments. Results are printed and saved under ignored `verification-output/`. Bootstrap intervals are not recomputed by this helper.

## 2. Manuscript figures, tables and secondary analyses

```sh
python -m pip install -r manuscript/requirements.txt
python manuscript/reproduce.py
```

This regenerates every figure (1–5 and S1), Tables 2–4 and S4–S5, the threshold-selection uncertainty analysis and the proposed uncertainty-aware threshold calibration. It uses only the saved per-image scores and estimates. It then checks that the regenerated analysis files and table bodies are byte-identical to those in `evidence/manuscript/`. Outputs are written to the ignored folder `manuscript-output/`. See [manuscript/README.md](../manuscript/README.md) for the script behind each item.

## 3. Initial development experiment (GPU/Colab)

Historical code is under `experiments/development/`. The corrected Colab notebook contains the original embedded scripts and setup. Outputs and execution counts have been cleared in the repository copy. The broad requirements file is not an exact environment lock. The recorded original evaluation environment is in `evidence/development/results/evaluation_provenance.json`; it includes Python 3.13.15, PyTorch 2.11.0+cu130, Transformers 4.57.1, NumPy 2.1.3, scikit-learn 1.7.2 and Pillow 11.3.0.

The script entry points are:

```sh
python experiments/development/prepare.py --data runs/reproduction/data --study runs/reproduction/study --download --test-source montgomery
python experiments/development/experiment.py encode --data runs/reproduction/data --study runs/reproduction/study --batch-size 4
python experiments/development/experiment.py fit --data runs/reproduction/data --study runs/reproduction/study
python experiments/development/experiment.py evaluate --data runs/reproduction/data --study runs/reproduction/study
```

These commands require the relevant ML packages, source downloads, substantial storage and GPU access for practical feature extraction. They have not been rerun during repository setup. Use a new output directory; do not overwrite the completed experiment. Compare generated source-image hashes, manifest and split with the archived manifest before claiming reproduction. The review file adjacent to `prepare.py` records AI-assisted image-similarity decisions, not expert radiological adjudication or confirmed patient identity.

The initial study manifest contains the 529/133 Shenzhen development partition and 138 Montgomery images. The RAD-DINO revision is pinned in the source. Training uses three seeded neural classifiers and a linear classifier. The final thesis's earlier Llama report evaluation and Kaggle TB classifier are separate from these journal experiments.

## 4. External and later experiments

The historical extension scripts keep their original absolute paths and hash checks. Their default shared root is `/content/drive/MyDrive/PulmoVision_Journal`.

| Stage | Historical folders beneath that root | Repository code |
|---|---|---|
| Initial development | `PV_classifier_v2_reviewed/study` | `experiments/development/` |
| TBX11K cohort audit | `TBX11K_audit_v1` | `experiments/external/audit_tbx11k.py` |
| Initial external evaluation | `TBX11K_evaluation_v1` | `experiments/external/evaluate_tbx11k.py` |
| Initial comparators | `Comparator_extension_v1` | `experiments/comparators/run_comparators.py`, `compare_saved_models.py` |
| Matched encoder–resolution experiment | `Matched_resolution_v1` | `experiments/matched_resolution/` |
| Preprocessing diagnostic | `DINO_preprocessing_audit_v1` | `experiments/preprocessing/` |

These runners also depend on source-archive backups and receipts, trained initial models, and `/content/TBX11K.zip`; those assets are not fully packaged here. Restore the original Drive archive and inspect each script's required paths before running. Do not bypass a failed hash check to make a run continue. A new environment or intentionally changed script should produce a separately versioned run.

`analyse_case_mix.py` additionally retains the original local project paths beginning `outputs/pulmovision-...`. It is archived as provenance and is not portable from its current repository location without explicitly adapting those paths. Historical protocol files may say execution was pending when they were written; the corresponding completion receipts document subsequent output.

## 5. What remains to make this a full execution archive

See [RELEASE_STATUS.md](RELEASE_STATUS.md). In particular, retrieve original trained weights, archive/download receipts, feature caches where needed, original comparator execution/environment records, and full matched-resolution/preprocessing per-image outputs. Confirm their hashes against the saved receipts. This repository currently supports full regeneration of the manuscript figures and tables from saved scores, saved point-estimate verification and inspection of the historical methods; it is not a guaranteed one-command reproduction of every experiment.
