# PulmoVision corrected exploratory classifier study

V1 is preserved. V2 changes duplicate grouping: exact image matches and inferred same-patient identifiers are grouped; dHash similarity alone does not group or exclude images. All 35 cross-source candidates were visually reviewed by Codex AI using 256-pixel contact sheets and judged to have distinct image content. This is not a clinical review or proof of distinct patient identity. No within-source candidate exceeded the audit correlation priority threshold of 0.98; remaining within-source candidates are retained as unreviewed screening evidence. Transformed duplicates can still escape detection.

The adjudicated_pairs.csv record provides reasons for each decision. Preparation verifies that the reviewed cross-source pair list exactly matches the current candidate list and saves its hash with the protocol. Final image counts are determined by the actual preparation output; the 800-row audit suggests no exclusions under this rule.

Use Shenzhen for development, a fixed grouped 80/20 split, and Montgomery as an exploratory held-out-source benchmark. Montgomery results have already been seen. New confirmatory claims need a further untouched source. Do not tune using Montgomery outcomes.

Keep the same frozen RAD-DINO checkpoint and deterministic processor, linear baseline, three neural-head seeds, 20-epoch schedule, validation AUROC checkpoint selection and validation-only 90%-sensitivity threshold rule as V1. No new calibration or threshold optimization is introduced. This isolates the split correction. Save validation predictions to investigate score and threshold stability. Report every seed, the 0.5-threshold metrics and 2,000-repetition group-bootstrap intervals; do not choose a winning seed on test results.

All evidence is saved to MyDrive/PulmoVision_Journal/PV_classifier_v2_reviewed. Existing models/results are never overwritten. Scripts and the Colab notebook include synthetic checks. This is a corrected exploratory baseline, not a completed journal validation or a reproduction of the thesis.
