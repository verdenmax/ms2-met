# Counterfactual hard-negative PSM construction

## Goal

Augment training with objectively wrong peptide hypotheses that are evaluated
against real light/heavy evidence. These rows are training-only Silver
negatives. They are not search-engine results, FDR decoys, or replacements for
the immutable real entrapment test set.

Feature CSV storage remains `label=1` for correct identifications and
`label=0` for incorrect identifications. Evaluation converts explicitly to
`error_truth = 1 - stored_label` and `error_score = 1 - trust_score`.

## Phase-1 scope

- SILAC only.
- Unmodified, prepared `label_type=positive` parent PSMs.
- Input raws are split before candidate generation.
- Parent and every child share `group_id` and `peptide_group_id`.
- Features are extracted from the original metabolically labelled raw with
  the existing single-raw (`feature_type=0`) workflow.
- Hardness is assigned only after real feature extraction.

`tools.counterfactual_parents` consumes an explicit confirmation table and a
complete raw-to-split manifest. It does not derive confirmation from signal
thresholds. Prepared JSON rows carry `heavy_confirmed=true`, `dataset_split`,
and a canonical `peptide_group_id`; their manifest and audit record the
versioned confirmation rule and input fingerprints. The negative generator
requires and validates this contract by default.

The repository's first executable 2Da pilot uses
`config/counterfactual/2da_label_dev_train.parents.ini`,
`2da_label_dev_train.negatives.ini`, and the standard run-directory config
`runs/counterfactual_2da_label_dev_train/config.ini`. The mass-spectrum config
inherits the nine raw paths, extraction settings, and speclib settings from
`runs/baseline_2da_clean/config.ini`; only the input PSM JSON and output feature
path change. Target and contaminant paths use the same values as
`extract_2da_pfind_diann.ini`. `make counterfactual-2da` owns the three-stage
hand-off without duplicating those paths. Its reviewed heavy-confirmation CSV
is a required untracked truth input and has no automatic producer.

Given a parent PSM `P`, a wrong hypothesis `Q` inherits `P`'s observed raw,
precursor m/z, RT, and charge. The sequence is replaced by `Q`; therefore
theoretical light fragments and all heavy precursor/fragment shifts are
recomputed from `Q` by the ordinary workflow.

## Candidate sources

### Composition shuffle

Shuffle all residues except the terminal cleavage residue. Composition and
precursor mass are retained; K/R positions may change.

### K/R-position shuffle

Shuffle only non-K/R residues. Composition, precursor mass, and the SILAC
fragment-shift signature are retained.

### Local mass-gap proposal

Replace a short internal segment with a different residue string whose total
mass is compatible with the observed precursor. Most parent fragments remain
shared, while a configured minimum number of positional fragments must remain
distinguishable.

Version 1 is not full spectrum-guided local de novo. It does not select two
observed fragment anchors or infer a gap directly from their measured mass.
The manifest records `local_uses_observed_fragment_anchors=false`. A later
implementation may add an observed-anchor proposal source behind the same
candidate-building interface.

## pReLoc-informed design constraints

The local report
[pReLoc.pdf](/home/verden/pfind/2026-fall/年中技术报告/pReLoc.pdf)
supports the general strategy of constructing strong alternatives in the
same observed context, but it addresses PTM-site localization in DDA rather
than peptide-identity confidence in DIA-SILAC. The detailed comparison and
page-level evidence are recorded in
[the pReLoc research note](../research/2026-09-01-preloc-negative-generation-and-training.md).

Three roles described by pReLoc must remain separate here:

- supervised competing sites improve a discrimination model;
- spectrum-guided local de novo candidates define a hard candidate space;
- backbone-preserving modification-transfer decoys estimate site-level FLR.

Counterfactual children in this project occupy only the first role. They are
Silver supervised negatives. They must not be counted as search decoys, used
to estimate FDR/FLR, or used to lock a production threshold. Conversely, a
future proposal generator may borrow pReLoc's local mass-graph idea without
adopting its FLR formula.

The transferable construction principle is to preserve all context outside
the error being modelled. For this project, that means preserving the raw,
observed precursor, RT, and charge while changing the peptide explanation as
locally as possible. Global composition and K/R-position shuffles remain
useful controls, but the primary hard-negative source should eventually be a
local alternative supported by observed fragment evidence.

Precursor compatibility remains a validity condition because `Q` is tested
against `P`'s observed precursor. Equal K/R count, identical K/R positions,
and equal label shift are not universal validity conditions. They are
candidate attributes used to define hardness strata; candidates whose label
shift resembles the parent are expected to be the most resistant to the
"heavy peak absent" shortcut.

## Validity versus hardness

Validity is established before feature extraction:

- the child differs from the parent after L/I normalization;
- it is absent as an exact or L/I-normalized target/contaminant substring;
- its theoretical precursor agrees with the inherited observed precursor
  within the configured tolerance;
- it retains enough theoretically distinguishable fragment positions;
- it is unique within its parent family.

The current target index does not compute full L2/L3 proteome-neighbour scans;
the audit states the actual exclusion scope as
`exact_or_li_substring_v1` rather than calling it full L4 classification.

Hardness is measured only after real light/heavy extraction. Candidate tiers
may include no-signal, partial-interference, high-interference, and OOF-mined
adversarial negatives. Generator metadata and provenance never enter model
features.

The realism audit must evaluate the high-score tail, not only overall class
separation. On a development split, each synthetic source/tier is compared
with real entrapment errors using grouped-OOF trust/error-score distributions,
KS distance, a source-classifier audit, and fixed-FPR missed-error metrics.
The immutable heldout set is not used to select a generator, tune a hardness
cutoff, or mine a negative.

## Interface and future replicate seam

`build_counterfactual_negatives(parents, target, config, contaminant)` is the
module interface. It returns PSM rows, a provenance manifest, and an audit;
file I/O is a CLI adapter.

Replicate support is intentionally deferred. Once phase 1 improves immutable
real-entrapment evaluation, partner-coordinate resolution becomes a real seam
with two adapters:

- metabolic shift: same raw, sequence-derived heavy m/z, approximately equal
  RT;
- replicate identity: different raw, unchanged peptide m/z, aligned RT.

Candidate generation and downstream relationship features remain unchanged.

## Planned observed-anchor source

The next local source is `synthetic_local_observed_anchor_v2`. It is added as
a new source rather than silently changing `local_mass_gap_v1` semantics. Its
implementation should:

1. select a short local window using observable light-side b/y evidence;
2. derive a measured mass gap between fragment/XIC anchors;
3. enumerate a small top-K of precursor-compatible alternative paths;
4. retain substantial shared evidence plus enough distinguishing fragments;
5. compute every candidate's fragment-specific heavy coordinates from its
   own K/R distribution; and
6. defer the final hardness label until ordinary light/heavy feature
   extraction and grouped-OOF mining.

pReLoc scores a top-150 peak graph from a relatively isolated DDA spectrum.
That implementation is not copied literally. The DIA adapter must instead
use fragment XIC coelution, mass error, and peak-shape evidence appropriate to
multiplexed DIA data. Candidate proposal should primarily use light-side
evidence; heavy-side agreement remains independent relationship evidence for
hardness measurement and training.

The outer `build_counterfactual_negatives(...)` module remains the stable
interface. Once v2 is implemented, theoretical mass-gap and observed-anchor
proposal implementations form a real internal seam. Raw/XIC access is
provided by an adapter at that seam; common validity, target exclusion,
grouping, provenance, and audit behaviour stay local to the outer module.

## Training progression

The first experiment retains the calibrated binary LightGBM objective. Family
metadata initially controls splitting, sampling, OOF mining, and diagnostics,
not the model input. Parent preparation now assigns `peptide_group_id` from
the L/I-normalized parent sequence so the same parent peptide cannot cross
folds through different charge states; `candidate_family_id` continues to
identify the narrower parent-hypothesis family. Formal training must select
`peptide_group_id` (or a connected grouping that contains it) as its split
group.

Only after an observed-anchor source improves real heldout errors should
training compare pairwise margins, family-softmax/listwise ranking, or an
InfoNCE auxiliary objective. Any family-aware model must still expose a
calibrated `trust_score = P(correct identification)` for a single PSM. Family
ranking accuracy is diagnostic and does not replace canonical fixed-FPR
evaluation.

pReLoc's ESM-2 sequence prior and Bayesian upstream peptide-score prior are
not adopted by default. In this task they can reveal generator membership or
copy a correct parent's confidence into a wrong child. They require separate
leakage-controlled ablations; search confidence and generator provenance
remain excluded from model features.

## Evaluation contract

Split raws before generation and keep the immutable real entrapment test free
of synthetic queries. Compare Gold-only training against Gold plus each
synthetic source and hardness tier, followed by a separate comparison of the
binary and any family-aware objectives. Report `roc_auc`, `error_pr_auc`,
`fnr_at_fpr5`, `error_recall_at_fpr10`, and applicable `fpr_1`, `fpr_5`, and
`fpr_10` working points under
`metric_semantics=error_identification_positive_v1`.
