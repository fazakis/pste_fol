# PSTE + Fast Outward Ladder oversampling algorithms repository

This folder is the standalone implementation repository for PSTE + FOL algoritm.
It contains:

- a standalone `PSTEClassifier` wrapper for tree classifiers;
- a standalone `FastOutwardLadderOversampler` implementation;
- runnable oversampler and imbalance-ensemble rival baselines;
- the 25 packaged datasets used in the manuscript experiments; and
- packaged reference result tables used to check the reported manuscript numbers.

## Repository layout

```text
pste_fol/
├── pste_fol/                      # standalone Python package
│   ├── pste.py                    # PSTEClassifier wrapper
│   ├── oversampling.py            # Fast Outward Ladder + rival oversamplers
│   ├── classifiers.py             # tree classifier/rival factories
│   ├── datasets.py                # packaged and CSV dataset loaders
│   ├── experiment.py              # CV experiment runner internals
│   └── metrics.py
├── scripts/
│   ├── run_experiment.py          # full/subset/custom experiment CLI
│   └── check_paper_alignment.py   # verifies packaged reference results
├── data/
│   ├── manifest.json              # dataset metadata/checksums
│   └── fast25/*.joblib            # all 25 manuscript datasets
├── reference/
│   ├── paper_results/*.csv        # fold-level manuscript reference results
│   └── paper_material/*.csv       # Friedman/Holm summary tables
└── outputs/                       # local generated results
```

## Install

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# optional editable install
pip install -e .
```

If you are inside the parent project environment, the existing `.venv` can also
run the scripts directly.

## Proposed methods

### Fast Outward Ladder (`FastOutwardLadderOversampler`)

`pste_fol/oversampling.py` implements Fast Outward Ladder as a standalone
oversampler with a `fit_resample(X, y)` interface. It creates a SMOTE seed cloud
and then adds outward ladder rungs that move away from nearby majority mass while
checking minority support, boundary ratio, local tube radius, and duplicates.

Example:

```python
from pste_fol.oversampling import FastOutwardLadderOversampler

sampler = FastOutwardLadderOversampler(sampling_strategy=1.0, random_state=42)
X_res, y_res = sampler.fit_resample(X_train, y_train)
```

### PSTE (`PSTEClassifier`)

`pste_fol/pste.py` implements PSTE as a wrapper around any scikit-learn-compatible
tree classifier. It accepts any oversampler name or object implementing
`fit_resample(X, y)`.

PSTE fits:

1. an original-prior branch on the original training fold;
2. a shadow branch on an oversampled copy of the same fold; and
3. a leakage-free nested inner-CV selector for the branch blend.

The selector searches shadow-score fraction `alpha` and blend mode under
constraints for predicted-prior drift, Brier degradation, and precision@top-k
loss. `alpha=1/3` corresponds to an approximate original:shadow ratio of `2:1`.

Example:

```python
from sklearn.ensemble import RandomForestClassifier
from pste_fol import PSTEClassifier

base = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
clf = PSTEClassifier(
    base,
    oversampler="fast_outward_ladder",
    sampling_strategy=1.0,
    total_estimators=200,
    random_state=42,
)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]
```

## Included datasets

`data/fast25/` contains all 25 binary imbalanced datasets used in the manuscript.
Class `1` is always the minority/positive class. The metadata and SHA-256 hashes
are in `data/manifest.json`.

Dataset shorthand for the CLI:

- `fast25`, `paper`, or `all`: all 25 packaged manuscript datasets;
- individual names such as `blood-transfusion`, `keel-yeast6`, etc.;
- a CSV path for a user dataset.

For a CSV dataset, pass `--target target_column`; if omitted, the last column is
used as the target. Binary labels are encoded with the minority class as `1`.

## Included methods/rivals

Tree classifiers:

- `rf`
- `extratrees`
- `bagged_cart`
- `random_subspace_trees`
- `decision_tree`

Oversamplers:

- `random_over_sampler`
- `smote`
- `smote_tomek`
- `kmeans_smote`
- `adasyn`
- `borderline_smote`
- `deep_smote` (standalone tabular DeepSMOTE-style latent implementation)
- `mgvae` (standalone majority-guided implementation)
- `fast_outward_ladder`

Imbalanced-ensemble rivals:

- `balanced_random_forest`
- `balanced_bagging`
- `easy_ensemble`
- `rus_boost`

## Run a quick smoke test

```bash
python scripts/run_experiment.py \
  --datasets blood-transfusion \
  --seeds 42 \
  --folds 2 \
  --classifiers rf \
  --oversamplers smote fast_outward_ladder \
  --method-groups native oversampler pste rivals \
  --total-estimators 20 \
  --pste-inner-cv-folds 2 \
  --pste-inner-cv-repeats 1 \
  --pste-alpha-grid 0,0.333333,0.5 \
  --pste-modes prob,priorcorr_logit \
  --output outputs/smoke.csv
```

## Reproduce the manuscript-style experiment

The manuscript experiment uses 25 datasets, 3 seeds, 5 folds, three tree
backbones, nine oversamplers, PSTE variants, the extra fixed-logit PSTE--FOL
variant, and four imbalance-ensemble rivals repeated in the same backbone-result
contexts as the paper reference files. A full rerun is computationally expensive
because PSTE performs nested inner validation inside every outer fold.

Use `--paper-exact` for the 27,000-row manuscript method menu and naming:

```bash
python scripts/run_experiment.py \
  --paper-exact \
  --datasets fast25 \
  --seeds 42 44 49 \
  --folds 5 \
  --classifier-n-jobs 1 \
  --output outputs/full_paper_rerun.csv
```

`--paper-exact` sets the paper method menu and PSTE defaults automatically:
`rf`, `extratrees`, `bagged_cart`; the nine manuscript oversamplers; `native`,
`oversampler`, `pste`, and `rivals`; 200 estimator budget; 3×2 PSTE inner CV;
continuous-alpha grid; and paper-compatible `fc_acc_ppob_*` method names.

You can reduce runtime by selecting datasets, classifiers, oversamplers, seeds,
or folds:

```bash
python scripts/run_experiment.py \
  --datasets blood-transfusion keel-glass0 \
  --seeds 42 \
  --folds 3 \
  --classifiers rf \
  --oversamplers fast_outward_ladder \
  --method-groups pste \
  --total-estimators 50 \
  --output outputs/pste_fol_subset.csv
```

Oversampler hyperparameters can be passed as JSON and are forwarded to named
oversampler constructors:

```bash
python scripts/run_experiment.py \
  --datasets blood-transfusion \
  --classifiers rf \
  --oversamplers fast_outward_ladder \
  --method-groups oversampler pste \
  --oversampler-kwargs '{"max_candidates": 4000, "candidate_multiplier": 4}' \
  --output outputs/fol_custom_kwargs.csv
```

## Run on your own dataset

```bash
python scripts/run_experiment.py \
  --datasets /path/to/my_data.csv \
  --target outcome \
  --seeds 42 \
  --folds 5 \
  --classifiers rf extratrees \
  --oversamplers fast_outward_ladder smote \
  --method-groups native oversampler pste \
  --output outputs/my_data_results.csv
```

## Check consistency with the manuscript results

The folder includes the fold-level reference CSVs and the Friedman/Holm material
used by the manuscript. Verify them with:

```bash
python scripts/check_paper_alignment.py
```

After a fresh full rerun, also verify the generated file has the same paper-exact
row/method/block signature:

```bash
python scripts/check_paper_alignment.py --generated outputs/full_paper_rerun.csv
```

Add `--compare-generated-values` only when you expect near bit-identical library
versions and want PR-AUC method means compared numerically.

Expected checks include:

- 27,000 reference rows;
- 25 datasets × 3 classifiers = 75 dataset-classifier blocks;
- PSTE + Fast Outward Ladder mean PR-AUC ≈ `0.6869525`;
- PSTE + Fast Outward Ladder vs sampler-only SMOTE: mean delta ≈ `0.0203595`,
  `68/75` wins;
- PSTE + Fast Outward Ladder vs PSTE + SMOTE: mean delta ≈ `0.0021240`,
  `51/75` wins, Holm `p≈0.0322775`.

## Notes on exact reproducibility

The packaged `reference/paper_results/*.csv` files are the authoritative
fold-level outputs used for the manuscript tables. A fresh rerun should be close
but may not be bit-identical across scikit-learn, imbalanced-learn, BLAS, and
CPU versions. The consistency script checks the reference outputs directly;
`--generated` checks a fresh rerun's paper-exact row/method/block shape; and
smoke runs verify that the standalone implementation and CLI execute end-to-end.

The standalone DeepSMOTE and MGVAE entries are self-contained tabular
implementations of the same rival families. They avoid external paper-code
repositories so the repo is runnable by itself; the packaged reference CSVs
remain the exact fold-level results used in the manuscript.
