# Nested-CV raw soft-voting control

- Scope: 18 numerical tasks, three backbones, seeds 42/44/49, five outer folds.
- Inner selection: three-fold stratified CV inside each outer-training fold.
- Candidates: raw probability-voting shadow weights 0.10 to 0.90 in steps of 0.05.
- Deployment budget: 100 original-prior trees plus 100 FOL-shadow trees.
- Complete outer rows: 810
- Complete prediction rows: 183321
- Recorded errors: 0
- The reference run was distributed only to reduce wall-clock time; the
  standalone runner is cluster-independent.
- Exact fixed-1:1 reproduction maximum absolute difference: 0

## Result

- Nested tuned raw mean task AP: 0.720932
- Selected alpha median/mode: 0.40/0.35
- Retained PSTE--FOL minus nested tuned raw: +0.002468
- Median difference: +0.002349
- W/T/L for retained: 16/0/2
- Matched rank-biserial correlation: 0.719298
- Task bootstrap 95% CI: [+0.000526, +0.004298]
- Exact Wilcoxon p: 0.0055999756
- Holm p in the five-test RQ2 architecture family: 0.0077209473
- Nested tuned raw minus fixed 1:1 raw: -0.000090
- Nested tuned raw minus fixed 2:1 raw: -0.001389
