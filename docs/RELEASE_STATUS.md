# Archive and release status

The repository is public and contains research code and saved evaluation evidence. Code is released under the MIT licence and derived records under CC BY 4.0. An archived release with a DOI will be deposited on Zenodo when the article is accepted.

| Component | Available here | Still required for a complete execution archive |
|---|---|---|
| Initial development | Scripts, notebooks, manifest, protocol, classifier metadata, training logs, validation scores and Montgomery scores/metrics | Trained `.pt`/`.npz` files, source images/archives and full feature/cache archive |
| TBX11K | Audit manifests, candidate/exclusion lists, cohort lock, predictions, metrics and completion receipt | Source images, dataset archive/receipt and full feature archive |
| Initial comparators | Execution and analysis scripts, aligned external predictions, summary/paired estimates, case-mix analysis and analysis receipt | Full training/validation records, trained weights, feature files and original execution/environment completion records |
| Matched resolution | Inference/analysis scripts, protocol, summary, paired contrasts, subgroup estimates and completion receipt | Full per-image predictions, fitted models/features and original inference receipt referenced by the analysis receipt |
| Preprocessing diagnostic | Script, protocol, replication summary, point estimates and completion receipt | Full per-image predictions/metrics, processor configuration, fitted models and features listed in that receipt |

The available receipts sometimes reference files absent from this checkout. The artifact verifier reports those omissions; it does not silently treat missing files as verified. Some original diagnostic plots are also absent from the local archive. Every figure and table in the manuscript can be regenerated with `manuscript/reproduce.py`.

Public-dataset identifiers and model outputs are included to support alignment and verification. Raw images and expert-review coordinator files are excluded. No human chest-radiograph review was performed. Image-similarity review must not be represented as clinical adjudication or a guarantee against patient overlap.

The listed gaps describe the current execution archive and should be considered when assessing reproducibility. The manuscript analyses in `manuscript/`, including the proposed uncertainty-aware threshold calibration, need only the saved per-image scores listed above.
