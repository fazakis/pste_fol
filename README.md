# PSTE + Fast Outward Ladder

Standalone implementation and reproducibility package for the retained
Prior–Shadow Tree Ensemble (PSTE) and Fast Outward Ladder-SMOTE (FOL)
algorithms described in:

> *Prior–Shadow Tree Ensembles for Imbalanced Tabular Classification*

The repository contains the current algorithm, the locked numerical benchmark,
the auxiliary nested-CV control, packaged datasets, fold-level reference
results, deterministic audits, and single-machine runners. A compute cluster
was used to shorten the original wall-clock time, but no cluster software or
infrastructure is required here.

## Current scope

The locked primary experiment contains:

- 18 numerical binary-classification tasks;
- Bagged CART, Random Forest, and ExtraTrees backbones;
- seeds 42, 44, and 49 with five outer folds;
- 22 logical methods, including eight standalone samplers and their eight
  matched PSTE wrappers;
- 17,820 fold–method rows; and
- task-level inference after averaging backbones, seeds, and folds.

Seven additional packaged datasets are retained as explicitly auxiliary data.
They are not silently mixed into the numerical primary analysis.

## Retained algorithms

### Fixed confidence-gated PSTE

PSTE trains two branches from the same outer-training fold:

1. an original-prior branch on the untouched training data;
2. a smaller shadow branch on an oversampled copy; and
3. a local, bounded correction applied only to the shadow score.

The retained configuration is fixed across tasks:

- 200 trees total, split into 133 original and 67 shadow trees;
- probability-space weights `2/3` and `1/3`;
- local prior neighbourhood `k=31`;
- prior smoothing `lambda=10`;
- local log-odds clip `[-2, 2]`;
- correction strength `gamma=0.15`;
- shadow-support neighbourhood `k_g=7`; and
- both-class confidence threshold `tau=3`.

For a query `x`, the corrected shadow probability and final score are:

```text
delta(x) = clip(logit(pi_o(x)) - logit(pi_s(x)), -2, 2)
q_s(x)   = sigmoid(logit(p_s(x)) + 0.15 * g(x) * delta(x))
S(x)     = (2/3) * p_o(x) + (1/3) * q_s(x)
```

The geometric support scale is learned once during `fit` from training-fold
queries and reused during prediction. Consequently, the score of one query is
invariant to unrelated rows placed in the same prediction batch; prediction
does not estimate any quantity from the test batch.

There is no inner validation, adaptive branch allocation, sampler selector, or
per-dataset tuning in the retained PSTE method. A logit blend remains available
only for custom API studies and is not part of the locked benchmark.

### Fast Outward Ladder-SMOTE

FOL fills the requested synthetic minority budget with ordinary local
interpolation plus accepted outward ladder points. Its directions point away
from an inverse-distance-weighted local majority centre. Candidate rungs use
fixed fractions `(0.12, 0.24, 0.38, 0.54)` and must pass feature-bound,
minority/majority distance, local-safety, and duplicate checks. Accepted
candidates are ranked once by a fixed deterministic heuristic; there is no
dataset-specific optimization.

The standalone FOL generator and the full PSTE prediction path reproduce the
canonical experiment implementation exactly in parity tests.

## Reference results

For the 18-task primary analysis:

- PSTE–FOL mean AP: `0.7233997190`;
- PSTE–FOL average rank: `4.2777778`, rank 1;
- all eight PSTE variants occupy the first eight ranks;
- Friedman `chi-square = 204.5928854`, `p = 4.5125e-32`;
- PSTE–FOL minus standalone FOL: mean AP difference `+0.0130618`,
  wins on `15/18` tasks, Holm `p = 0.0029526`; and
- the matched PSTE-family improvement is `+0.0189493`, with `18/18` wins.

The reviewer-motivated auxiliary control tunes an ordinary raw FOL soft vote
inside every outer-training fold. Its mean AP is `0.7209320`. Retained PSTE–FOL
is higher by `+0.0024677`, wins `16/18` tasks, and has Holm `p = 0.0077209`
within the five-test RQ2 mechanism family. This control is not a 23rd method in
the locked primary Friedman ranking.

## Repository layout

```text
pste_fol/
├── pste_fol/                         # algorithms and experiment library
├── scripts/
│   ├── run_experiment.py             # primary/custom single-machine runner
│   ├── run_primary_shards.py          # resumable 18-task primary runner
│   ├── run_nested_raw_vote_control.py
│   ├── analyze_nested_raw_vote_control.py
│   ├── fetch_mgvae.py
│   ├── check_data_manifest.py
│   ├── check_reference.py
│   └── build_reference_material.py
├── data/
│   ├── bestangle25/*.joblib          # 18 primary + 7 auxiliary datasets
│   └── manifest.json                 # shapes, scope, and SHA-256 hashes
├── reference/
│   ├── primary/                      # locked 18-task × 22-method results
│   ├── auxiliary_nested_raw_vote/    # nested-control metrics/predictions
│   └── REFERENCE_MANIFEST.json
├── tests/
└── outputs/                          # generated locally; ignored by Git
```

## Installation

Python 3.10 or newer is required. The exact validated base stack is pinned in
`requirements.txt`.

```bash
git clone https://github.com/fazakis/pste_fol.git
cd pste_fol
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The primary 22-method rerun also needs PyTorch and the original MGVAE source:

```bash
python -m pip install -r requirements-paper.txt
python scripts/fetch_mgvae.py
```

`fetch_mgvae.py` checks out
[`Aiqz/MGVAE`](https://github.com/Aiqz/MGVAE) at the exact benchmark revision
`cad386bd2b3a90f6b740cbdf5f0cec8834102ea5`. The upstream project is kept
external rather than vendored so its provenance and upstream terms remain
explicit. Set `PSTE_FOL_MGVAE_ROOT` only if that pinned checkout is stored
elsewhere.

## Direct API use

```python
from sklearn.ensemble import RandomForestClassifier
from pste_fol import PSTEClassifier

base = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=1,
)

model = PSTEClassifier(
    base,
    oversampler="fast_outward_ladder",
    total_estimators=200,
    random_state=42,
    n_jobs=1,
)
model.fit(X_train, y_train)
probability = model.predict_proba(X_test)[:, 1]
```

Class `1` must be the minority/positive class. `predict_components(X)` exposes
the original score, raw and corrected shadow scores, local shift, support gate,
class-confidence gate, combined gate, and final score.

Use FOL alone:

```python
from pste_fol import FastOutwardLadderOversampler

sampler = FastOutwardLadderOversampler(
    sampling_strategy=1.0,
    random_state=42,
)
X_shadow, y_shadow = sampler.fit_resample(X_train, y_train)
```

## Quick smoke test

This small run excludes MGVAE and finishes quickly:

```bash
python scripts/run_experiment.py \
  --datasets blood-transfusion \
  --seeds 42 \
  --folds 2 \
  --classifiers rf \
  --oversamplers smote fast_outward_ladder \
  --method-groups native oversampler pste \
  --total-estimators 20 \
  --output outputs/smoke.csv
```

With a 20-tree custom budget, PSTE keeps the fixed allocation rule and uses
13 original-prior plus 7 shadow trees.

## Reproduce the locked primary experiment

After installing the paper dependencies and fetching MGVAE, the recommended
single-machine command checkpoints one validated CSV per task and resumes from
those checkpoints:

```bash
python scripts/run_primary_shards.py \
  --classifier-n-jobs 1 \
  --mgvae-device auto \
  --output-dir outputs/kbs_primary_sharded
```

After all 18 shards pass their structural audits, the script creates
`outputs/kbs_primary_sharded/kbs_primary_17820.csv`. It locks the 18 tasks,
22 methods, three backbones, 200-tree budgets, fixed PSTE constants, exact
200-epoch MGVAE configuration, and probability blending. MGVAE is generated
once per outer fold and shared across its standalone/PSTE roles and downstream
backbones, matching the reference protocol.

The equivalent monolithic command is:

```bash
python scripts/run_experiment.py \
  --paper-exact \
  --classifier-n-jobs 1 \
  --mgvae-device auto \
  --output outputs/kbs_primary_17820.csv
```

The runner is cluster-independent and can execute sequentially on one machine.
The full rerun is computationally substantial—expect hours to days depending on
CPU/GPU hardware. A cluster changes wall-clock time only, not the protocol. A
sidecar `*.manifest.json` records the protocol, dependency versions, Git state,
resolved runtime/CUDA device, data-manifest hash, output size, and output
SHA-256.

Audit the generated method/fold signature, and optionally the task means:

```bash
python scripts/check_reference.py \
  --generated outputs/kbs_primary_sharded/kbs_primary_17820.csv

python scripts/check_reference.py \
  --generated outputs/kbs_primary_sharded/kbs_primary_17820.csv \
  --compare-generated-values \
  --generated-value-tol 1e-6
```

The value comparison assumes the pinned software stack. Small floating-point
differences may occur on a different numerical stack or architecture.

## Reproduce the nested raw-vote control

```bash
python scripts/run_nested_raw_vote_control.py \
  --paper-exact \
  --n-jobs 1 \
  --classifier-n-jobs 1 \
  --output-dir outputs/nested_raw_vote

python scripts/analyze_nested_raw_vote_control.py \
  --run-dir outputs/nested_raw_vote
```

The exact run produces 810 outer-fold rows and 183,321 held-out predictions.
Inside each outer-training fold, three-fold CV selects a raw shadow weight from
`0.10` to `0.90` in steps of `0.05`. Each deployment pair contains 100 original
and 100 FOL-shadow trees and has no PSTE correction. The analyzer reconstructs
every selected weight, probability, metric, task contrast, exact signed-rank
test, Holm adjustment, and reference comparison. The run directory also
contains `run_manifest.json` with the same environment and artifact provenance.

## Validate the packaged release

```bash
python scripts/check_data_manifest.py
python -m unittest discover -s tests -p 'test*.py' -v
python scripts/check_reference.py
```

Rebuild the dependence-safe task-level tables into a new local directory:

```bash
python scripts/build_reference_material.py \
  --metrics reference/primary/validated_fold_metrics_17820.csv \
  --output outputs/rebuilt_reference
```

`scripts/check_paper_alignment.py` is retained only as a compatibility alias
for `scripts/check_reference.py`.

## Dataset safety and custom data

The numerical primary task names are listed as `primary_numeric18` in
`data/manifest.json`; the remaining seven are listed as
`auxiliary_category7`. Every packaged file is covered by a SHA-256 hash.

For a custom numerical CSV, class labels are encoded so the smaller class is
class `1`:

```bash
python scripts/run_experiment.py \
  --datasets /path/to/data.csv \
  --target outcome \
  --seeds 42 \
  --folds 5 \
  --classifiers rf extratrees \
  --oversamplers fast_outward_ladder smote \
  --method-groups native oversampler pste \
  --output outputs/custom.csv
```

Categorical predictors are rejected by default because interpolation over
arbitrary ordinal codes is not semantically valid. The
`--allow-categorical-encoding` switch exists only for explicitly auxiliary
experiments and should not be interpreted as a category-safe PSTE/FOL method.

## Reference provenance

`reference/REFERENCE_MANIFEST.json` records every packaged reference artifact,
byte count, SHA-256 hash, primary protocol, inference unit, and exact MGVAE
revision. The reference directory includes:

- all 17,820 validated primary fold–method rows;
- the 18 × 22 task matrix and complete rankings;
- focused, matched-family, and all-pair exact statistical comparisons;
- per-backbone dataset mean/SD matrices and all-metric summaries; and
- all 810 nested-control rows plus 183,321 held-out branch predictions.

## Citation and license

Citation metadata is provided in `CITATION.cff`. The standalone PSTE/FOL code
in this repository is released under the MIT License. Fetched third-party code
and Python dependencies remain subject to their own licenses.
