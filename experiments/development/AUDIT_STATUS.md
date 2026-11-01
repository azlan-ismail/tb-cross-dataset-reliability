# Corrected study status — 24 September 2026

AI contact-sheet review of all 35 cross-source pairs completed; all show distinct image content. Pair-level reasons are saved in adjudicated_pairs.csv and in the Google Drive audit folder. No clinical or patient-identity review is claimed. Other within-source screening candidates remain unreviewed.

The corrected grouping preserves dHash flags as candidates without merging them. Exact matches and inferred same-patient records still group together. Preparation validates the reviewed candidate list and stores a hash of the decisions. All five synthetic pipeline checks passed in the corrected Colab notebook, including the similarity-only regression and the training/evaluation smoke test.

Corrected notebook: https://colab.research.google.com/drive/1DyAigBulxkbEvhnv-4JN1USXmxFWQtnu

T4 runtime allocated. Drive connected after retry. The original source endpoint returned an HTML error instead of the Montgomery ZIP; verified archives from the audit runtime were backed up to Drive and are being restored in the GPU runtime. Preparation, training and evaluation have been queued after recovery. Fresh corrected model training and evaluation are not yet complete. Original v1 evidence is preserved. Model settings and validation-only threshold selection are unchanged; validation predictions are now exported. Montgomery remains an exploratory benchmark, not a new untouched test.
