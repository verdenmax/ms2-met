# Project Instructions

## Identification-confidence evaluation semantics

All training, evaluation, reporting, plotting, and documentation work must
preserve the following distinction between the storage convention and the
statistical evaluation convention.

### Storage and model convention

- Feature CSVs store `label=1` for a correct identification and `label=0` for
  an incorrect identification.
- A model's native output is a trust/support score:
  `trust_score = P(correct identification)`; a larger value means more support.
- Do not invert the stored labels or retrain existing models solely to change
  metric semantics. A label migration is allowed only as an explicit,
  repository-wide schema migration.

### Canonical evaluation convention

- The actual positive class is an **incorrect identification**.
- The actual negative class is a **correct identification**.
- Convert explicitly at the evaluation boundary:

  ```text
  error_truth = 1 - stored_label
  error_score = 1 - trust_score
  ```

- A predicted positive means that an identification is flagged as incorrect
  or suspicious.
- FP: an actually correct identification is incorrectly flagged as wrong.
- FN: an actually incorrect identification is incorrectly accepted as correct.
- `FPR = FP / (FP + TN)` is the false-alarm rate among correct IDs.
- `FNR = FN / (TP + FN)` is the missed-error rate among incorrect IDs.
- Error recall is `TP / (TP + FN) = 1 - FNR`.

### Implementation and output requirements

- Reuse the canonical helpers in `tools/spec_trainer/src/cv_core.py`; do not
  create a second confusion-matrix or fixed-FPR implementation with different
  semantics.
- Decision thresholds used in evaluation are error-score thresholds:
  `error_score >= error_threshold` means incorrect/suspicious. If a trust-score
  threshold is also reported, name it explicitly as `trust_threshold`.
- Fixed-FPR working points must calibrate FPR on actually correct IDs and then
  report missed/detected incorrect IDs.
- Prefer unambiguous names such as `n_actual_correct`, `n_actual_error`,
  `error_pr_auc`, `error_recall`, and `correct_recall`. Do not introduce generic
  metric names such as `n_pos`, `n_neg`, `pos_recall`, or `neg_recall`.
- Machine-readable evaluation results must include:

  ```text
  metric_semantics = error_identification_positive_v1
  positive_class = incorrect_identification
  ```

- The standard reported metrics are `roc_auc`, `error_pr_auc`,
  `fnr_at_fpr5`, `error_recall_at_fpr10`, and the `fpr_1`, `fpr_5`, and
  `fpr_10` working points when applicable.
- ROC plots, PR metrics, confusion matrices, rescore tables, ablation results,
  cross-test results, logs, tests, and prose must all use the same convention.

### Historical-result compatibility

- A result file without
  `metric_semantics=error_identification_positive_v1` is a legacy result.
- Do not make a legacy result look current by renaming its keys. Rerun the
  evaluation or training pipeline before comparing fixed-FPR metrics.
- Complementing both labels and scores leaves ROC-AUC numerically unchanged,
  and legacy `pr_auc_neg` may equal the new `error_pr_auc`; this numerical fact
  does not validate legacy FNR or working-point fields.
- Any change to metric code must include value-level tests that pin the FP, FN,
  FPR, and FNR definitions above.
