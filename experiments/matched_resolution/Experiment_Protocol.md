# Matched-resolution experiment

Prepared 25 September 2026. Status: scripts uploaded and execution cell saved in the existing TBX11K Colab notebook; execution awaits successful Google Drive mounting.

## Question

Does RAD-DINO retain an advantage over DINOv2 when both use identical image preprocessing and the same input resolution? Does increasing input resolution improve either encoder?

## Four conditions

| Encoder | Input sizes | Adaptation |
|---|---|---|
| RAD-DINO | 224×224 and 518×518 | Frozen CLS features, balanced logistic regression |
| DINOv2 | 224×224 and 518×518 | Frozen CLS features, balanced logistic regression |

Both use EXIF orientation correction, RGB conversion, bicubic shortest-edge resize to the selected size, centre crop, and ImageNet channel normalization. Encoder inference uses float32, evaluation mode, eager attention and batch size 4. No augmentation or encoder fine-tuning is performed. Each condition has its own train-only StandardScaler and logistic classifier (C=1, maximum 2,000 iterations, seed 42).

Pinned checkpoints: microsoft/rad-dino at 110cbc18d5133582e320b43d53bf5c44e410c936; facebook/dinov2-base at f9e44c814b77203eaa57a6bdbbd535f21ede1415.

## Data and selection

The existing manifests are hash-checked. Shenzhen supplies 529 training and 133 validation images. Montgomery supplies 138 external images. The locked TBX11K validation-derived external cohort contains 1,794 images: 200 TB, 800 healthy and 794 other-disease images. Every source image is checked against its recorded hash before preprocessing.

For each condition, select the threshold on Shenzhen validation to attain sensitivity of at least 90%, then minimize false-positive rate, then choose the highest eligible threshold. Apply this fixed threshold to both external cohorts. External labels never enter fitting or threshold selection.

## Analysis

Report AUROC, average precision, sensitivity, specificity, Brier score and confusion counts. Compute 95% percentile intervals for AUROC, sensitivity, specificity and Brier score with 10,000 shared, unstratified image bootstrap draws, seed 20260925. Pairing is maintained across models. Compare RAD-DINO minus DINOv2 at each resolution, 518 minus 224 within each encoder, and the change in the encoder gap between resolutions. Provide healthy-versus-other-disease subgroup point estimates using the same TB-positive cases.

The report and figure are generated automatically after all four conditions finish. Outputs are saved under MyDrive/PulmoVision_Journal/Matched_resolution_v1. Feature extraction checkpoints every 128 images. Completed outputs are verified and preserved on restart.

## Interpretation boundaries

This extension was chosen after earlier external results were observed and is exploratory. Comparisons are not multiplicity-adjusted. Bootstrap uncertainty is conditional on the fitted models and fixed split; it does not capture training-data variation. Matching preprocessing controls this specific confound but does not isolate a causal effect of medical pretraining. Patient/source independence and pretraining dataset exposure remain unverified. There is no expert clinical review.

## Execution

Open the existing Colab notebook (not included in this archive; `run_matched_resolution.py` contains the experiment code).

Run its setup cell, complete the Google Drive connection, and run the final cell beginning “Matched-resolution experiment.” Do not run all cells: earlier experiments have already completed.

Local checks: both execution scripts parse successfully; the saved development manifest has 800 unique image IDs and the expected 529/133/138 split. GPU inference and statistical self-checks still require successful Colab execution.
