# Synthetic hard-negative training-set construction

## Goal

Train an independent per-PSM score answering whether metabolically labelled
companion evidence supports the current identification. Synthetic negatives
augment training only; they are not FDR decoys and never replace the immutable
real entrapment test set.

## Deep module and external search seam

`tools.training_set_builder` is the module. Its interface has two phases:

1. `generate_queries(config)` writes a query manifest and FASTA.
2. `assemble_training_set(config)` consumes ordinary ms2-met feature rows
   after the FASTA has been searched externally.

DIA-NN/pFind-DIA is outside the module at the seam between those phases. The
module does not own search-engine binaries, raw files, search parameters, or
FDR estimation.

## Invariants

- Split complete raw runs into train/validation/test before query generation.
- Positive parents are independently light+heavy-confirmed.
- The immutable test search is never expanded with synthetic queries.
- Every training row is an independently scorable PSM; no candidate group is
  required at inference.
- `group_id` is split metadata only. A synthetic negative shares its parent
  positive's group to prevent cross-fold leakage.
- Provenance columns are excluded from model features.
- Synthetic q-values/counts are not interpreted as FDR.
- Query generation and assembly must use the same canonical labeling value:
  `silac`, `c13`, or `n15`.
- Uniform C13/N15 currently supports unmodified peptides only. Modified rows
  fail explicitly because PTM C/N atoms are not represented by the mass model.

## Query generators

### Controlled shuffle

Non-K/R residues are shuffled while K/R cleavage positions remain fixed.
This generator happens to preserve composition, precursor mass, and label
shift exactly, but exact per-parent matching is not a global requirement.

### Proteome-like Markov

A first- or second-order residue model is learned from the target FASTA.
Generated peptides have a valid tryptic terminus and are accepted when their
precursor m/z and label shift occupy the configured parent bins. This enforces
distribution overlap without requiring identical K/R or C/N composition.

Both generators exclude exact target peptides, L/I isomers, contaminants,
duplicates, and candidates too close to their parent sequence.

## Silver physical-signal gate

After external search and normal ms2-met extraction, a Silver candidate must:

- have non-zero light and heavy precursor peaks;
- have at least the configured number of light fragments;
- have at least the configured number of shifted/co-eluting heavy fragments;
- be in the acquired heavy range;
- still classify as L4 against target and contaminant FASTAs.

At least one explicit acquisition-range field (`heavy_in_raw` or
`heavy_out_of_range`) is required. Its absence is a schema error, not evidence
that the candidate was acquired.

Candidates are capped per generator inside positive-distribution bins. The
audit JSON reports filter counts, binning columns, standardized mean
differences, and an OOF metadata-only generator-signature AUC.
The canonical distribution feature is `total_label_shift`. Legacy
`total_silac_shift` is accepted only for SILAC tables; pre-fix C13/N15 values
are rejected because they were calculated with SILAC chemistry.

For uniform C13/N15, the current natural-abundance isotope-envelope feature is
not scientifically calibrated without labeling enrichment/purity. Feature
extraction therefore writes `isotope_correlation=NaN` and
`isotope_model_valid=0` rather than treating a placeholder zero as evidence.

## Output labels

| `negative_source` | label | confidence |
|---|---:|---|
| `gold_positive` | 1 | gold |
| `gold_entrapment` | 0 | gold |
| `silver_synthetic_shuffle` | 0 | silver |
| `silver_synthetic_markov` | 0 | silver |

Training may sample 1:1 batches from this table, but the table itself and the
immutable test set retain their available natural row counts.

## Commands

```bash
cp training_set_builder.ini.example training_set_builder.ini
python -m tools.training_set_builder generate --config training_set_builder.ini

# Search datasets/synthetic_queries.fasta only against TRAIN raw files,
# convert reported candidates to the usual ms2-met features.csv.

python -m tools.training_set_builder assemble --config training_set_builder.ini
```

For CV training, use `data.group_col: group_id`. Mixed-label groups are
supported: a parent positive and its derived Silver negatives remain in the
same fold.
