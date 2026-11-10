# Licence for derived evaluation records

The derived evaluation records in this repository are licensed under the
[Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

This covers the records produced by the authors, including:

- `evidence/` (manifests, per-image scores, estimates, logs and completion receipts, and the analysis outputs in `evidence/analysis/`);
- `validation/` (recomputed point metrics);
- `SOURCE_MAP.csv` and `ARTIFACTS.sha256`.

Source code (`experiments/`, `analysis/` and `scripts/`)
is licensed under the MIT licence in [LICENSE](LICENSE).

## Material from the source datasets

The CC BY 4.0 licence above covers only the authors' own derived records, such
as model scores, image hashes, analyses and estimates. It does not apply to
the image file names, identifiers and diagnostic labels, which come from the
source datasets and remain subject to their providers' terms:

- **Shenzhen and Montgomery chest X-ray sets** are provided by the U.S.
  National Library of Medicine. Obtain them from NLM
  (<https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/>)
  and cite Jaeger et al. (2014), *Quantitative Imaging in Medicine and
  Surgery* 4(6):475-477.
- **TBX11K** is provided by the Media Computing Lab, Nankai University, under
  CC BY 4.0 (<https://mmcheng.net/tb/>). Cite Liu et al. (2020),
  *Rethinking Computer-aided Tuberculosis Diagnosis*, CVPR 2020.

The radiographs themselves are **not** included in this repository. Pretrained
model weights are not redistributed and remain subject to their own licences.

Please cite the archive using [CITATION.cff](CITATION.cff).
