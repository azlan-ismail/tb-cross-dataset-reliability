RESEARCH FIGURE RESOURCES

CONTENTS
data/: 42 unrounded plotted estimates, saved confidence limits and provenance.
evidence/: unchanged copies of ten source files used for plots or protocol context.
SOURCE_AUDIT.json: source-file SHA-256 hashes and data checks.
PLOT_CHECK.json: plotted estimates, interval types and canvas bounds checks.
captions.txt: descriptions of the external-reliability and encoder/preprocessing plots.
make_figures.py and requirements.txt: portable plotting code and dependency.

RECONSTRUCTION
From the repository root:
  python -m pip install -r figures/requirements.txt
  python figures/make_figures.py

The script reads the included CSV files and exports PDF, PNG and SVG plots
under figures/figures/. No model fitting, GPU, resampling or source images
are required. The filenames retain the original figure2/figure3 identifiers.

INTERPRETATION AND SCOPE
Figure 2 A/B: saved 95% bootstrap intervals; Montgomery 2,000 resamples and TBX11K
10,000 resamples. The sensitivity reference is the development selection target.
Figure 2 C: saved Wilson 95% intervals for subgroup specificity. Seed results are
separate fitted models and are not pooled.
Figure 3 A/B: 10,000 shared within-cohort bootstrap resamples. B shows saved paired
AUROC differences, rather than using overlap of marginal intervals as a test.
Figure 3 C: six post hoc point estimates with no confidence intervals. All three
pipelines produce 224-by-224 inputs, regenerate both development and external
features, and refit the classifier. No new statistical inference was performed.

The figures visualize the archived estimates. They do not
establish patient-level or pretraining independence, clinical validity, a causal
medical-pretraining effect, or an optimal preprocessing procedure. All comparisons
are exploratory. The initial models in Figure 2 are distinct fitted pipelines from
the later matched conditions and diagnostic conditions in Figure 3.

