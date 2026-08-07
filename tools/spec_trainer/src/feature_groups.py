"""Scientific feature registry for SILAC-DIA confidence experiments.

This module is the single source of truth for two separate decisions:

* whether a CSV column may be used as a model input; and
* which scientific evidence family a permitted input belongs to.

The registry is deliberately explicit.  Prefix rules are attractive here but
unsafe: ``precursor_centering`` is acquisition context whereas
``precursor_light_centering_defect`` is an observed MS1 chromatographic
feature.  Every model input therefore has exactly one registered role.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


# Identification, labels, provenance, or candidate descriptors that are kept
# in features.csv but are not model inputs.
METADATA_COLUMNS = frozenset({
    "sequence", "charge", "raw_title1", "raw_title2", "labeling",
    "protein_names", "label", "label_type",
    "precursor_mz", "sequence_len", "rt",
    "negative_source", "negative_confidence", "query_id", "parent_id",
    "group_id", "generator", "generator_seed", "heavy_confirmed",
})


# Columns excluded from training because they are search-confidence fields,
# known constants, or strong acquisition-condition proxies.  They are still
# registered so a raw features.csv header can be audited exhaustively.
TRAINING_EXCLUDED_COLUMNS = frozenset({
    "modification_count",
    "window_width",
    "fragment_xic_empty_count",
    "fragment_same_mass_count",
    "fragment_heavy_absent_count",
    "q_value",
})


# Availability/validity flags define whether an evidence family is evaluable.
# They should be used for cohort filtering and missingness reporting, not as
# classifier inputs in the formal ablation arms.
ELIGIBILITY_FEATURES = frozenset({
    "heavy_in_raw",
    "heavy_out_of_range",
    "precursor_xic_empty",
    "q1a_valid",
    "has_lib_pred",
    "isotope_model_valid",
})


# Candidate/acquisition descriptors that contain no measured light-heavy
# agreement. Counts here quantify evidence opportunity, not successful match.
CONTEXT_FEATURES = frozenset({
    "kr_count",
    "total_label_shift",
    "total_silac_shift",  # legacy alias; removed when canonical is present
    "valid_fragment_ions_num",
    "precursor_centering",
    "psm_is_split_window",
})


# Directly observed precursor-level light/heavy evidence from MS1 XICs and
# isotope/mass relationships.  This intentionally excludes the DIA-window
# coordinate ``precursor_centering``.
MS1_OBSERVED_FEATURES = frozenset({
    "precursor_pearson",
    "precursor_apex_delta",
    "precursor_apex_delta_signed",
    "precursor_mz_avg_err",
    "precursor_light_max_int",
    "precursor_heavy_max_int",
    "precursor_intensity_ratio",
    "precursor_cosine",
    "precursor_snr",
    "precursor_peak_width_ratio",
    "precursor_peak_symmetry",
    "precursor_light_apex_cycle_offset",
    "precursor_light_apex_cycle_offset_signed",
    "precursor_heavy_apex_cycle_offset",
    "precursor_heavy_apex_cycle_offset_signed",
    "precursor_base_to_apex_ratio",
    "precursor_n_peaks",
    "precursor_smoothness",
    "isotope_correlation",
    "mass_shift_error",
    "precursor_light_centering_defect",
    "precursor_heavy_centering_defect",
    "precursor_light_shape_irregularity",
    "precursor_heavy_shape_irregularity",
    "precursor_light_base_to_apex_ratio",
    "precursor_light_n_peaks",
    "precursor_light_smoothness",
    "precursor_light_narrow_defect",
    "precursor_heavy_narrow_defect",
})


# Directly observed fragment-level light/heavy evidence from MS2 XICs.  The
# q1a TP/FN and recall fields compare observed signals against the
# sequence-specific shifted/unshifted fragment pattern.
MS2_OBSERVED_FEATURES = frozenset({
    "b_count", "b_p25", "b_p50", "b_p75", "b_mean", "b_std",
    "b_min", "b_high_ratio",
    "y_count", "y_p25", "y_p50", "y_p75", "y_mean", "y_std",
    "y_min", "y_high_ratio",
    "all_count", "all_p25", "all_p50", "all_p75", "all_mean",
    "all_std", "all_min", "all_high_ratio",
    "frag_corr_weighted", "matched_intensity_percent",
    "all_apex_delta_mean", "all_apex_delta_p50",
    "all_apex_delta_std", "all_apex_delta_max",
    "all_mz_err_mean", "all_mz_err_p50", "all_mz_err_std",
    "all_mz_err_max",
    "all_cosine_mean", "all_cosine_p50", "all_cosine_std",
    "all_cosine_max",
    "all_snr_mean", "all_snr_p50", "all_snr_std", "all_snr_max",
    "all_light_apex_cycle_offset_mean",
    "all_light_apex_cycle_offset_p50",
    "all_light_apex_cycle_offset_std",
    "all_light_apex_cycle_offset_max",
    "all_light_apex_cycle_offset_signed_mean",
    "all_light_apex_cycle_offset_signed_p50",
    "all_light_apex_cycle_offset_signed_std",
    "all_light_apex_cycle_offset_signed_max",
    "all_heavy_apex_cycle_offset_mean",
    "all_heavy_apex_cycle_offset_p50",
    "all_heavy_apex_cycle_offset_std",
    "all_heavy_apex_cycle_offset_max",
    "all_heavy_apex_cycle_offset_signed_mean",
    "all_heavy_apex_cycle_offset_signed_p50",
    "all_heavy_apex_cycle_offset_signed_std",
    "all_heavy_apex_cycle_offset_signed_max",
    "all_log_lh_ratio_std", "all_log_lh_ratio_mad",
    "b_log_lh_ratio_std", "b_log_lh_ratio_mad",
    "y_log_lh_ratio_std", "y_log_lh_ratio_mad",
    "all_base_to_apex_ratio_mean", "all_base_to_apex_ratio_p50",
    "all_base_to_apex_ratio_std", "all_base_to_apex_ratio_max",
    "all_n_peaks_mean", "all_n_peaks_p50", "all_n_peaks_std",
    "all_n_peaks_max",
    "all_smoothness_mean", "all_smoothness_p50", "all_smoothness_std",
    "all_smoothness_max",
    "all_heavy_centering_defect_mean",
    "all_heavy_centering_defect_max",
    "all_heavy_shape_irregularity_mean",
    "all_heavy_shape_irregularity_max",
    "all_heavy_narrow_defect_mean", "all_heavy_narrow_defect_max",
    "all_light_centering_defect_mean", "all_light_centering_defect_max",
    "all_light_shape_irregularity_mean",
    "all_light_shape_irregularity_max",
    "all_light_base_to_apex_ratio_mean",
    "all_light_base_to_apex_ratio_max",
    "all_light_n_peaks_mean", "all_light_n_peaks_max",
    "all_light_smoothness_mean", "all_light_smoothness_max",
    "all_light_narrow_defect_mean", "all_light_narrow_defect_max",
    "q1a_recall", "q1a_recall_shifted",
    "q1a_recall_unshifted_separable", "q1a_y_recall", "q1a_b_recall",
    "q1a_TP_count", "q1a_FN_count", "q1a_TP_shifted",
    "q1a_TP_unshifted_separable",
    "q1a_total_count",
})


# Features that consume spectral-library predictions. ``has_lib_pred`` is an
# eligibility flag; ``n_fragments_in_F`` is prediction-conditioned opportunity
# and therefore stays in this group rather than acquisition context.
MS2_PREDICTED_FEATURES = frozenset({
    "spec_pattern_SA_b", "spec_pattern_SA_y", "spec_pattern_SA",
    "spec_pattern_spearman_b", "spec_pattern_spearman_y",
    "spec_pattern_spearman", "spec_pattern_LH_consistency",
    "pred_hl_ratio_cv", "pred_hl_ratio_mad",
    "pred_coverage", "pred_coverage_wpred",
    "unexpected_heavy_fraction", "unexpected_heavy_intensity_ratio",
    "pred_both_present_fraction", "n_both_present", "global_lh_ratio",
    "pred_coverage_adaptive", "pred_coverage_coelut",
    "frag_offtime_fraction", "spec_pattern_SA_coelut",
    "n_fragments_in_F",
})


FEATURE_GROUPS = {
    "eligibility": ELIGIBILITY_FEATURES,
    "context": CONTEXT_FEATURES,
    "ms1_observed": MS1_OBSERVED_FEATURES,
    "ms2_observed": MS2_OBSERVED_FEATURES,
    "ms2_predicted": MS2_PREDICTED_FEATURES,
}


# Formal ablation arms. Eligibility flags are deliberately absent: experiments
# must construct a common evaluable cohort before resolving an arm.
EXPERIMENT_ARMS = {
    "context_only": ("context",),
    "ms1_only": ("ms1_observed",),
    "ms2_observed_only": ("ms2_observed",),
    "ms2_all": ("ms2_observed", "ms2_predicted"),
    "ms1_ms2_no_prediction": ("ms1_observed", "ms2_observed"),
    "evidence_all": (
        "ms1_observed", "ms2_observed", "ms2_predicted"),
    "full": ("context", "ms1_observed", "ms2_observed",
             "ms2_predicted"),
}


@dataclass(frozen=True)
class FeatureRegistryAudit:
    """Observable result of checking one features.csv schema."""

    unregistered_columns: tuple[str, ...]
    ungrouped_model_columns: tuple[str, ...]
    missing_registered_columns: tuple[str, ...]
    group_counts: dict[str, int]

    @property
    def is_complete(self) -> bool:
        return not self.unregistered_columns and not self.ungrouped_model_columns


@dataclass(frozen=True)
class FeatureSpec:
    """Stable semantic identity for a physical CSV column."""

    physical_name: str
    canonical_name: str
    group: str | None
    model_input: bool


def _registered_columns() -> frozenset[str]:
    columns = set(METADATA_COLUMNS) | set(TRAINING_EXCLUDED_COLUMNS)
    for features in FEATURE_GROUPS.values():
        overlap = columns.intersection(features)
        if overlap:
            raise RuntimeError(
                f"feature registry has duplicate assignments: {sorted(overlap)}")
        columns.update(features)
    return frozenset(columns)


REGISTERED_COLUMNS = _registered_columns()


_CANONICAL_NAME_OVERRIDES = {
    "precursor_centering": "context_light_precursor_dia_window_position",
    "valid_fragment_ions_num": "ms2_attempted_fragment_pair_count",
    "total_label_shift": "context_total_label_mass_shift_da",
    "total_silac_shift": "context_total_label_mass_shift_da",
    "precursor_mz_avg_err": "ms1_heavy_mean_signed_ppm_error",
    "mass_shift_error": "ms1_heavy_apex_signed_ppm_error",
    "isotope_correlation": "ms1_heavy_isotope_envelope_cosine",
    "precursor_intensity_ratio": "ms1_light_to_heavy_xic_area_ratio",
    "precursor_snr": "ms1_heavy_p25_floor_snr",
    "precursor_peak_symmetry": "ms1_heavy_peak_area_asymmetry",
    "matched_intensity_percent": "ms2_matched_fragment_intensity_fraction",
    "pred_hl_ratio_cv": "ms2_pred_weighted_log10_hl_std",
    "pred_hl_ratio_mad": "ms2_pred_log10_hl_mad",
    "global_lh_ratio": "ms2_pred_global_heavy_to_light_ratio",
    "n_fragments_in_F": "ms2_pred_topk_fragment_count",
}


def _default_canonical_name(column: str) -> str:
    name = _CANONICAL_NAME_OVERRIDES.get(column, column)
    name = name.replace("spec_pattern_SA", "spec_pattern_spectral_angle")
    name = name.replace("spec_pattern_LH", "spec_pattern_lh")
    name = name.replace("q1a_TP", "q1a_tp").replace("q1a_FN", "q1a_fn")
    if name.endswith("smoothness"):
        name = name[:-len("smoothness")] + "roughness"
    name = name.replace("_smoothness_", "_roughness_")
    return name


CANONICAL_FEATURE_NAMES = {
    column: _default_canonical_name(column)
    for column in REGISTERED_COLUMNS
}


def canonical_feature_name(column: str) -> str:
    """Resolve a stable scientific name without renaming stored CSVs."""
    try:
        return CANONICAL_FEATURE_NAMES[column]
    except KeyError as exc:
        raise KeyError(f"unregistered feature column: {column!r}") from exc


def feature_spec(column: str) -> FeatureSpec:
    """Return the canonical name, group and training role for one column."""
    canonical = canonical_feature_name(column)
    group = next(
        (name for name, features in FEATURE_GROUPS.items()
         if column in features),
        None,
    )
    return FeatureSpec(
        physical_name=column,
        canonical_name=canonical,
        group=group,
        model_input=group is not None,
    )


def audit_feature_registry(columns: Iterable[str]) -> FeatureRegistryAudit:
    """Audit an available CSV schema without reading data rows.

    Schema variants may omit registered columns, so omissions are reported but
    are not considered incomplete. Unknown columns and otherwise-trainable
    columns without a scientific group are considered incomplete.
    """
    available = tuple(dict.fromkeys(columns))
    available_set = set(available)
    model_columns = (
        available_set - set(METADATA_COLUMNS) - set(TRAINING_EXCLUDED_COLUMNS))
    grouped = set().union(*FEATURE_GROUPS.values())
    return FeatureRegistryAudit(
        unregistered_columns=tuple(sorted(available_set - REGISTERED_COLUMNS)),
        ungrouped_model_columns=tuple(sorted(model_columns - grouped)),
        missing_registered_columns=tuple(sorted(REGISTERED_COLUMNS - available_set)),
        group_counts={
            name: len(available_set.intersection(features))
            for name, features in FEATURE_GROUPS.items()
        },
    )


def resolve_experiment_arm(
        arm: str, available_columns: Sequence[str], *, strict: bool = True,
) -> list[str]:
    """Return one formal ablation arm in input-header order.

    ``strict=True`` rejects schema drift before training. When both shift names
    occur, only the canonical ``total_label_shift`` is returned.
    """
    if arm not in EXPERIMENT_ARMS:
        raise ValueError(
            f"unknown experiment arm {arm!r}; expected one of "
            f"{sorted(EXPERIMENT_ARMS)}")

    audit = audit_feature_registry(available_columns)
    if strict and not audit.is_complete:
        raise ValueError(
            "feature schema is not fully registered: "
            f"unregistered={list(audit.unregistered_columns)}, "
            f"ungrouped_model={list(audit.ungrouped_model_columns)}")

    selected_groups = EXPERIMENT_ARMS[arm]
    allowed = set().union(*(FEATURE_GROUPS[name] for name in selected_groups))
    result = [column for column in available_columns if column in allowed]
    if "total_label_shift" in result and "total_silac_shift" in result:
        result.remove("total_silac_shift")
    if not result:
        raise ValueError(f"experiment arm {arm!r} resolved to zero features")
    return result
