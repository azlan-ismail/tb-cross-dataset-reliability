# TBX11K source audit

24 September 2026. Status: computational audit and priority visual comparison complete. An exploratory evaluation cohort of 1,794 images is frozen. No model inference performed.

## Results and decision

- All 1,800 source-specific validation images were readable and consistent with the filename list and annotation dimensions.
- No exact file/pixel matches were found against the 800 original Montgomery/Shenzhen images.
- Six pairs within the candidate cohort had identical grayscale pixels. One image per pair was retained using lexicographic image-ID order, without reference to model predictions. All six repeated images were in the sick/non-TB group; there were no conflicting labels in these groups.
- The retained cohort contains 200 TB-labelled, 800 healthy and 794 sick/non-TB images (1,794 total).
- All 111 priority pairs were inspected by Codex AI across 19 contact sheets. The displayed rib/clavicle geometry and internal contour/texture patterns differed; none was identified as a copied image at the 256-pixel viewing resolution. This is duplicate-content screening, not clinical or patient-identity adjudication.
- The remaining 5,218 lower-priority similarity pairs were not individually visually reviewed. Within-cohort transformed duplicates and foundation-model exposure were not exhaustively checked. The cohort is suitable for a transparently limited exploratory benchmark, not a certified leakage-free clinical validation.

The TB-labelled images comprise 157 with annotation category 1 only, 36 with category 2 only, and seven with both. The optional category-1 secondary analysis therefore has 164 positive images, excluding the 36 category-2-only images rather than relabelling them negative.

The candidate archive SHA-256 is `890d309b7ecaead989f527ff6eb756194cd89200cbe71c7287d1895ef0b2beab`. The retained manifest SHA-256 is `629461acc1b839740264e0ba0a17b7e2baf15201b187a4b2da16a6544d3e6d6c`. The manifest, six exclusions, 111 pair decisions and limitations are saved in the `tbx-audit-evidence` directory and in Drive. The next experiment must use the existing four frozen models and thresholds without tuning to these outcomes.

## Source and release

The official author repository links to Drive file `1r-oNYTPiPCOUzSjChjCIYTdkjBTugqxR`. Drive metadata reports TBX11K.zip, 3,306,757,175 bytes, modified 24 June 2020. The downloaded file matches that byte count and opens as a ZIP. The archive hash is recorded by the audit. [Official repository](https://github.com/yun-liu/Tuberculosis).

The authors' project page specifies CC BY 4.0 and explains that released images are 512 × 512, despite higher-resolution originals. This release resolution is a limitation for the external benchmark. Attribution must accompany published results. [Author project page](https://mmcheng.net/tb/).

## Confirmed archive findings

The archive README explicitly states that combined `all_*` lists mix TBX11K with DA, DB, Montgomery and Shenzhen. Those combined lists are unsuitable as an independent cohort for the existing models.

The actual `TBX11K_val.txt` contains 1,800 images: 800 `health/`, 800 `sick/` and 200 `tb/`. Only this source-specific list is used. Its JSON annotations are checked against the full filename set, not joined by row order.

The three annotation names are ActiveTuberculosis, ObsoletePulmonaryTuberculosis and PulmonaryTuberculosis. The category-2 name differs from the paper's “latent” wording. We retain the exact released terminology and treat the primary endpoint as a dataset-label benchmark, not confirmed active infection. Empty detection annotations do not by themselves define a negative image: the healthy/sick group and TB group are retained explicitly and consistency-checked.

Released image metadata examples provide an internal image ID, filename, dimensions, capture timestamp and licence field, without a patient ID or institution field. Internal IDs and export-like timestamps are not evidence of patient identity or hospital independence. The paper's statement of one image per individual is documented separately; unknown patient identities remain unknown.

## Computation

For all 1,800 validation images, record file and grayscale pixel hashes, dimensions, dHash and pHash. Compare against the 800 original study images, reading the verified Montgomery and Shenzhen archives. Check exact within-cohort duplicates too.

Cross-source candidates are the union of exact matches, dHash distance ≤4, pHash distance ≤6, or 64 × 64 normalised grayscale correlation ≥0.95. Correlation also checks a horizontal reflection. Save the nearest reference for every candidate image. Priority review includes exact matches, correlation ≥0.95, or pHash distance ≤6 with correlation ≥0.90. Candidate flags are not automatic duplicate exclusions. Lower-priority flags remain recorded and unresolved until reviewed; screening cannot guarantee detection of cropped, rotated or other transformed copies.

No weights, thresholds or model choices change during this audit. No clinical reviewer is available, so any image comparison is limited to duplicate-content assessment and must not be described as clinical adjudication. Foundation-model image exposure remains a separate unresolved question.

[Saved audit evidence in Drive](https://drive.google.com/drive/folders/1DaJMXzBzhoEBC1DJhgS83vvW_aUIOxgD)
