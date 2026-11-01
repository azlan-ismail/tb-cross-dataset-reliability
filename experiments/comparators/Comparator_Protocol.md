# Exploratory comparator extension v1

Specified 24 September 2026 after inspecting RAD-DINO's external results. No prospective or untouched-test claim is made for this extension.

## Fixed experiments

- Frozen facebook/dinov2-base CLS features with its official processor, train-only StandardScaler, and balanced logistic regression (C=1, maximum 2,000 iterations). The exact repository revision is resolved and saved before feature extraction.
- Torchvision DenseNet121 initialized with IMAGENET1K_V1, fully fine-tuned for 20 epochs with seeds 42, 43 and 44. Official deterministic resize/central-crop/normalization transforms, no augmentation. AdamW learning rate 0.0001, weight decay 0.0001, batch eight, cosine schedule and inverse-frequency weighted cross-entropy. Select earliest highest Shenzhen validation AUROC per seed.
- The original 529/133 Shenzhen training/validation split and the same 138 Montgomery and 1,794 TBX11K images. All image bytes must match the saved manifests. No external image or label enters model fitting or threshold selection.
- Select thresholds at validation sensitivity at least 90%, then maximum specificity, then highest threshold for ties. Freeze before inference. Keep all three seeds; do not select the best external result.

DenseNet is a conventional fine-tuned comparator, whereas DINOv2 and RAD-DINO are frozen representations. Their preprocessing and adaptation regimes differ. These comparisons measure pipeline performance and do not isolate architecture or medical pretraining as a causal factor. A matched-input/optimization ablation would be needed for a narrower causal claim.

## Outputs and recovery

Save protocol, software versions, exact DINOv2 revision, image-processor configuration, validation predictions, training logs, selected model weights, thresholds, external per-image scores, point metrics and integrity hashes in Comparator_extension_v1. Save DenseNet optimizer, scheduler and random states after every epoch. Preserve previously completed external runs. A changed code/environment must not silently resume an earlier protocol.

The runner produces point estimates first. Subsequent analysis should verify alignment and hashes and calculate paired model comparisons and intervals on the shared images, without further fitting. Primary reporting includes both external cohorts and healthy versus other-illness specificity. Report exploratory multiplicity and image-level independence limitations.

## Completed post hoc case-mix analysis

Using the original RAD-DINO predictions, compare the same 200 TB images against 800 healthy negatives, 794 other-illness negatives, and both combined. Five thousand paired stratified bootstrap resamples (seed 20260925) share TB draws and preserve each stratum's sample size. Healthy-minus-other-illness AUROC differences are positive for all four classifiers. This post hoc analysis changes no model, threshold or original prediction. See Case_Mix_Analysis.md and Case_Mix_Analysis.json.

## References

- [DINOv2 model and processor](https://huggingface.co/facebook/dinov2-base)
- [Torchvision DenseNet121 weights and transforms](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html)
